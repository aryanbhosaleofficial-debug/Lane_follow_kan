"""Camera intrinsics: parameters container and chessboard-based calibration.

Conventions
-----------
* Pixel coordinates follow OpenCV: x right, y down, and the *centre* of the top-left pixel is
  (0, 0). The image therefore spans [-0.5, width-0.5] x [-0.5, height-0.5].
* Chessboard size is the number of INNER corners (columns, rows), not the number of squares.
* The physical square size must be given together with an explicit unit; it is stored in metres.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ._util import (UNDISTORT_EPSILON, UNDISTORT_ITERATIONS, UNIT_TO_METERS, array_to_list, as_points,
                    frozen_array, is_positive_finite, is_positive_int, is_real_number)
from .errors import (CalibrationQualityError, InsufficientCalibrationDataError,
                     InvalidCalibrationInputError)

log = logging.getLogger(__name__)

#: model name -> number of distortion coefficients
DISTORTION_MODELS: dict[str, int] = {"radial2": 5, "standard5": 5, "rational8": 8, "fisheye": 4}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, UNDISTORT_ITERATIONS, UNDISTORT_EPSILON)


def _undistort_pinhole_points(src: np.ndarray, K: np.ndarray, D: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Iterative point undistortion with explicit convergence criteria.

    OpenCV 4.x exposes the criteria through ``undistortPointsIter``; OpenCV 5.x removed that function and added the
    ``criteria`` argument to ``undistortPoints``. The default criteria leave ~1 px of round-trip error under strong
    distortion, so they are always overridden.
    """
    if hasattr(cv2, "undistortPointsIter"):
        return cv2.undistortPointsIter(src, K, D, None, P, _UNDISTORT_CRITERIA)
    return cv2.undistortPoints(src, K, D, None, None, P, _UNDISTORT_CRITERIA)


def _fisheye_flag(name: str) -> int:
    """Fisheye calibration flags live in ``cv2.fisheye`` (OpenCV 4.x) or in ``cv2`` itself (5.x)."""
    for owner in (cv2.fisheye, cv2):
        if hasattr(owner, name):
            return getattr(owner, name)
    raise AttributeError(f"OpenCV {cv2.__version__} has no fisheye flag {name}")


# --------------------------------------------------------------------------------------------
# Quality report
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class IntrinsicQuality:
    """Outcome of the quality checks on an intrinsic calibration.

    ``status`` is ``"ok"``, ``"warning"`` or ``"poor"``. The checks compare against thresholds the
    caller configured; they do not certify geometric accuracy on a real camera.
    """
    status: str
    issues: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    flagged_views: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"status": self.status, "issues": list(self.issues), "metrics": dict(self.metrics),
                "flagged_views": list(self.flagged_views)}

    @classmethod
    def from_dict(cls, data: dict) -> "IntrinsicQuality":
        if not isinstance(data, dict) or data.get("status") not in ("ok", "warning", "poor"):
            raise InvalidCalibrationInputError("quality.status must be ok, warning or poor")
        metrics = data.get("metrics", {})
        if not isinstance(metrics, dict) or not all(is_real_number(v) and math.isfinite(v) for v in metrics.values()):
            raise InvalidCalibrationInputError("quality.metrics must map names to finite numbers")
        return cls(data["status"], tuple(str(i) for i in data.get("issues", [])),
                   {str(k): float(v) for k, v in metrics.items()},
                   tuple(str(v) for v in data.get("flagged_views", [])))


# --------------------------------------------------------------------------------------------
# Intrinsic parameters
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class IntrinsicParameters:
    """Camera matrix and distortion coefficients for one image resolution."""
    image_width: int
    image_height: int
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    model: str
    rms_reprojection_px: float | None = None
    per_view_rms_px: tuple[float, ...] = ()
    num_views: int = 0
    pattern_size: tuple[int, int] | None = None
    square_size_m: float | None = None
    quality: IntrinsicQuality | None = None

    def __post_init__(self):
        if not is_positive_int(self.image_width) or not is_positive_int(self.image_height):
            raise InvalidCalibrationInputError("image size must be positive integers")
        object.__setattr__(self, "image_width", int(self.image_width))
        object.__setattr__(self, "image_height", int(self.image_height))
        if self.model not in DISTORTION_MODELS:
            raise InvalidCalibrationInputError(
                f"unknown distortion model {self.model!r}; choose one of {sorted(DISTORTION_MODELS)}")
        K = frozen_array(self.camera_matrix, (3, 3), "camera_matrix")
        D = frozen_array(np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1), None, "dist_coeffs")
        if D.shape != (DISTORTION_MODELS[self.model],):
            raise InvalidCalibrationInputError(
                f"model {self.model!r} needs {DISTORTION_MODELS[self.model]} distortion coefficients, got {D.shape[0]}")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise InvalidCalibrationInputError("focal lengths must be positive")
        if not np.array_equal(K[2], [0.0, 0.0, 1.0]) or K[1, 0] != 0 or abs(K[0, 1]) > 1e-6 * K[0, 0]:
            raise InvalidCalibrationInputError("camera matrix must have the form [[fx,0,cx],[0,fy,cy],[0,0,1]] (zero skew)")
        if not (0 <= K[0, 2] <= self.image_width and 0 <= K[1, 2] <= self.image_height):
            raise InvalidCalibrationInputError("principal point lies outside the image")
        if np.abs(D).max() > 1e4:
            raise InvalidCalibrationInputError("distortion coefficients are implausibly large")
        object.__setattr__(self, "camera_matrix", K)
        object.__setattr__(self, "dist_coeffs", D)
        if self.rms_reprojection_px is not None and not (is_real_number(self.rms_reprojection_px)
                                                         and math.isfinite(self.rms_reprojection_px)
                                                         and self.rms_reprojection_px >= 0):
            raise InvalidCalibrationInputError("rms_reprojection_px must be a finite non-negative number")
        object.__setattr__(self, "per_view_rms_px", tuple(float(v) for v in self.per_view_rms_px))
        if not all(math.isfinite(v) and v >= 0 for v in self.per_view_rms_px):
            raise InvalidCalibrationInputError("per-view errors must be finite and non-negative")
        if self.square_size_m is not None and not is_positive_finite(self.square_size_m):
            raise InvalidCalibrationInputError("square_size_m must be positive and finite")

    @property
    def image_size(self) -> tuple[int, int]:
        """(width, height)."""
        return (self.image_width, self.image_height)

    @property
    def is_fisheye(self) -> bool:
        return self.model == "fisheye"

    # -- point transforms ------------------------------------------------------------------
    def undistort_points(self, points_px: Any, new_camera_matrix: np.ndarray | None = None) -> np.ndarray:
        """Raw (distorted) pixel coordinates -> ideal pinhole pixels in ``new_camera_matrix``.

        Non-finite input rows yield NaN rows. Accuracy is iterative (round trip < 1e-6 px for
        points inside the model's invertible region).
        """
        pts = as_points(points_px, allow_nonfinite=True)
        P = self.camera_matrix if new_camera_matrix is None else np.asarray(new_camera_matrix, dtype=np.float64)
        out = np.full_like(pts, np.nan)
        ok = np.isfinite(pts).all(axis=1)
        if ok.any():
            src = pts[ok].reshape(-1, 1, 2)
            if self.is_fisheye:
                res = cv2.fisheye.undistortPoints(src, self.camera_matrix, self.dist_coeffs, None, None, P, _UNDISTORT_CRITERIA)
            else:
                res = _undistort_pinhole_points(src, self.camera_matrix, self.dist_coeffs, P)
            out[ok] = res.reshape(-1, 2)
        return out

    def project_normalized(self, normalized_xy: Any) -> np.ndarray:
        """Normalised ideal coordinates (x/z, y/z) -> raw distorted pixel coordinates."""
        xn = as_points(normalized_xy, "normalized_xy", allow_nonfinite=True)
        out = np.full_like(xn, np.nan)
        ok = np.isfinite(xn).all(axis=1)
        if ok.any():
            obj = np.column_stack([xn[ok], np.ones(int(ok.sum()))])
            zero = np.zeros(3)
            if self.is_fisheye:
                res, _ = cv2.fisheye.projectPoints(obj.reshape(1, -1, 3), zero, zero, self.camera_matrix, self.dist_coeffs)
            else:
                res, _ = cv2.projectPoints(obj, zero, zero, self.camera_matrix, self.dist_coeffs)
            out[ok] = res.reshape(-1, 2)
        return out

    def distort_points(self, undistorted_px: Any, new_camera_matrix: np.ndarray) -> np.ndarray:
        """Ideal pixels (in ``new_camera_matrix``) -> raw distorted pixel coordinates."""
        pts = as_points(undistorted_px, "undistorted_px", allow_nonfinite=True)
        Kn = frozen_array(new_camera_matrix, (3, 3), "new_camera_matrix")
        homog = np.column_stack([pts, np.ones(len(pts))])
        norm = homog @ np.linalg.inv(Kn).T
        return self.project_normalized(norm[:, :2])

    # -- resizing ------------------------------------------------------------------------
    def scaled_to(self, width: int, height: int, *, aspect_tolerance: float = 1e-3) -> "IntrinsicParameters":
        """Return parameters for a uniformly rescaled image of the same field of view.

        This is only geometrically valid when the new size is the same sensor region scaled
        uniformly (no crop, no different sensor mode). Aspect-ratio changes are rejected. The
        caller is responsible for knowing the camera does not crop in the new mode.
        """
        if not is_positive_int(width) or not is_positive_int(height):
            raise InvalidCalibrationInputError("target size must be positive integers")
        sx, sy = width / self.image_width, height / self.image_height
        if abs(sx - sy) > aspect_tolerance * max(sx, sy):
            raise InvalidCalibrationInputError(
                f"cannot rescale {self.image_width}x{self.image_height} to {width}x{height}: aspect ratio differs "
                f"(a crop or different sensor mode is not a scale); recalibrate at the new resolution")
        s = (sx + sy) / 2
        K = np.array(self.camera_matrix)
        K[0, 0] *= s; K[1, 1] *= s
        K[0, 2] = s * (K[0, 2] + 0.5) - 0.5   # pixel centres sit at integer coordinates
        K[1, 2] = s * (K[1, 2] + 0.5) - 0.5
        return IntrinsicParameters(width, height, K, self.dist_coeffs, self.model, self.rms_reprojection_px,
                                   self.per_view_rms_px, self.num_views, self.pattern_size, self.square_size_m,
                                   self.quality)

    # -- serialisation -------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "image_width": self.image_width, "image_height": self.image_height, "model": self.model,
            "camera_matrix": array_to_list(self.camera_matrix), "dist_coeffs": array_to_list(self.dist_coeffs),
            "rms_reprojection_px": self.rms_reprojection_px, "per_view_rms_px": list(self.per_view_rms_px),
            "num_views": self.num_views,
            "pattern_size": None if self.pattern_size is None else list(self.pattern_size),
            "square_size_m": self.square_size_m,
            "quality": None if self.quality is None else self.quality.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IntrinsicParameters":
        if not isinstance(data, dict):
            raise InvalidCalibrationInputError("intrinsics must be an object")
        missing = [k for k in ("image_width", "image_height", "model", "camera_matrix", "dist_coeffs") if k not in data]
        if missing:
            raise InvalidCalibrationInputError(f"intrinsics missing fields: {missing}")
        pattern = data.get("pattern_size")
        if pattern is not None and (not isinstance(pattern, (list, tuple)) or len(pattern) != 2):
            raise InvalidCalibrationInputError("pattern_size must be [columns, rows]")
        return cls(data["image_width"], data["image_height"], data["camera_matrix"], data["dist_coeffs"], data["model"],
                   data.get("rms_reprojection_px"), tuple(data.get("per_view_rms_px", ())), int(data.get("num_views", 0)),
                   None if pattern is None else (int(pattern[0]), int(pattern[1])), data.get("square_size_m"),
                   None if data.get("quality") is None else IntrinsicQuality.from_dict(data["quality"]))


# --------------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class IntrinsicCalibrationConfig:
    """Settings for chessboard intrinsic calibration.

    ``pattern_size`` is (columns, rows) of INNER corners. ``square_size`` needs an explicit
    ``square_unit`` (``m``, ``cm`` or ``mm``). The quality thresholds are caller-chosen screening
    limits, not accuracy guarantees: a low reprojection error does not prove a good calibration
    (it can hide poor view diversity), and a modest one does not prove a bad one.
    """
    pattern_size: tuple[int, int]
    square_size: float
    square_unit: str
    model: str = "standard5"
    min_views: int = 10
    detector: str = "classic"            # "classic" (findChessboardCorners + cornerSubPix) or "sb"
    subpixel_window: int = 11
    expected_image_size: tuple[int, int] | None = None   # (width, height) the camera is meant to run at
    max_rms_px: float = 1.0
    max_view_rms_px: float = 2.0
    min_coverage_fraction: float = 0.35  # fraction of a 8x6 image grid touched by detected corners
    min_orientation_spread_deg: float = 20.0   # max angle between any two board normals
    min_view_area_fraction: float = 0.02       # per-view corner hull area / image area (warning only)
    max_aspect_deviation: float = 0.05         # |fx/fy - 1| (warning only)
    enforce_quality: bool = True

    def __post_init__(self):
        ps = self.pattern_size
        if (not isinstance(ps, (tuple, list)) or len(ps) != 2 or not all(is_positive_int(v) for v in ps)
                or min(ps) < 3):
            raise InvalidCalibrationInputError(
                "pattern_size must be (columns, rows) of INNER corners, each >= 3 (a board with 10x7 squares has 9x6 inner corners)")
        object.__setattr__(self, "pattern_size", (int(ps[0]), int(ps[1])))
        if not is_positive_finite(self.square_size):
            raise InvalidCalibrationInputError("square_size must be positive and finite")
        if self.square_unit not in UNIT_TO_METERS:
            raise InvalidCalibrationInputError(f"square_unit must be one of {sorted(UNIT_TO_METERS)}")
        if self.model not in DISTORTION_MODELS:
            raise InvalidCalibrationInputError(f"model must be one of {sorted(DISTORTION_MODELS)}")
        if not is_positive_int(self.min_views) or self.min_views < 3:
            raise InvalidCalibrationInputError("min_views must be an integer >= 3")
        if self.detector not in ("classic", "sb"):
            raise InvalidCalibrationInputError("detector must be 'classic' or 'sb'")
        if not is_positive_int(self.subpixel_window):
            raise InvalidCalibrationInputError("subpixel_window must be a positive integer")
        if self.expected_image_size is not None and not (len(self.expected_image_size) == 2
                                                         and all(is_positive_int(v) for v in self.expected_image_size)):
            raise InvalidCalibrationInputError("expected_image_size must be (width, height) positive integers")
        for name in ("max_rms_px", "max_view_rms_px", "min_orientation_spread_deg", "max_aspect_deviation"):
            if not is_positive_finite(getattr(self, name)):
                raise InvalidCalibrationInputError(f"{name} must be positive and finite")
        for name in ("min_coverage_fraction", "min_view_area_fraction"):
            v = getattr(self, name)
            if not (is_real_number(v) and 0 <= v <= 1):
                raise InvalidCalibrationInputError(f"{name} must be within [0, 1]")

    @property
    def square_size_m(self) -> float:
        return float(self.square_size) * UNIT_TO_METERS[self.square_unit]


@dataclass(frozen=True)
class ViewRecord:
    """Per-image outcome. ``reason`` explains a rejection."""
    name: str
    used: bool
    reason: str = ""
    image_size: tuple[int, int] | None = None
    rms_px: float | None = None
    tilt_deg: float | None = None


@dataclass(frozen=True, eq=False)
class IntrinsicCalibrationResult:
    intrinsics: IntrinsicParameters
    quality: IntrinsicQuality
    views: tuple[ViewRecord, ...]
    corners: tuple[np.ndarray, ...]       # detected (N, 2) corner sets that were used
    rvecs: tuple[np.ndarray, ...]
    tvecs: tuple[np.ndarray, ...]

    def report(self) -> str:
        i, q = self.intrinsics, self.quality
        lines = [f"Intrinsic calibration: {q.status.upper()}  model={i.model}  size={i.image_width}x{i.image_height}",
                 f"  views used {i.num_views}/{len(self.views)}   overall RMS {i.rms_reprojection_px:.4f} px",
                 f"  fx={i.camera_matrix[0, 0]:.2f} fy={i.camera_matrix[1, 1]:.2f} cx={i.camera_matrix[0, 2]:.2f} cy={i.camera_matrix[1, 2]:.2f}",
                 f"  distortion {np.array2string(i.dist_coeffs, precision=5)}"]
        lines += [f"  metric {k} = {v:.4g}" for k, v in q.metrics.items()]
        lines += [f"  ISSUE: {m}" for m in q.issues]
        lines += [f"  rejected {v.name}: {v.reason}" for v in self.views if not v.used]
        return "\n".join(lines)


def chessboard_object_points(pattern_size: tuple[int, int], square_size_m: float) -> np.ndarray:
    """Board corner coordinates in the board frame (metres), row-major like OpenCV's detector.

    Index i = row * columns + column maps to (column * s, row * s, 0).
    """
    cols, rows = pattern_size
    grid = np.zeros((rows * cols, 3), np.float64)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m
    return grid


def load_gray_image(path: Path) -> np.ndarray | None:
    """Read an image as 8-bit grayscale; tolerant of non-ASCII Windows paths. None if unreadable."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except OSError:
        return None


def lens_invertibility_error(intrinsics: IntrinsicParameters) -> float:
    """Worst round-trip error (px) of raw pixel -> ideal -> raw over a grid covering the whole image.

    Near zero means every pixel the camera can deliver has a well-defined ideal ray. A large value means the distortion
    polynomial folds back or has no preimage inside the image, so undistortion (and any BEV) would be wrong there.
    """
    width, height = intrinsics.image_size
    us, vs = np.meshgrid(np.linspace(0, width - 1, 33), np.linspace(0, height - 1, 25))
    raw = np.column_stack([us.ravel(), vs.ravel()])
    und = intrinsics.undistort_points(raw, intrinsics.camera_matrix)
    err = np.linalg.norm(intrinsics.distort_points(und, intrinsics.camera_matrix) - raw, axis=1)
    return float(np.nanmax(err)) if np.isfinite(err).any() else float("inf")


class IntrinsicCalibrator:
    """Chessboard intrinsic calibration with explicit quality gating."""

    def __init__(self, config: IntrinsicCalibrationConfig):
        self.config = config

    # -- detection -----------------------------------------------------------------------
    def detect_corners(self, gray: np.ndarray) -> np.ndarray | None:
        """Return refined (N, 2) float64 inner corners, or None if the board is not found."""
        cfg = self.config
        if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
            raise InvalidCalibrationInputError("detect_corners needs a 2-D uint8 grayscale image")
        if cfg.detector == "sb":
            found, corners = cv2.findChessboardCornersSB(gray, cfg.pattern_size, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            return corners.reshape(-1, 2).astype(np.float64) if found else None
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, cfg.pattern_size, flags)
        if not found:
            return None
        half = max(2, cfg.subpixel_window // 2)
        refined = cv2.cornerSubPix(gray, corners.astype(np.float32), (half, half), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 1e-4))
        return refined.reshape(-1, 2).astype(np.float64)

    def calibrate_directory(self, directory: str | Path) -> IntrinsicCalibrationResult:
        """Detect the board in every image in ``directory`` (non-recursive) and calibrate."""
        folder = Path(directory)
        if not folder.is_dir():
            raise InvalidCalibrationInputError(f"calibration image directory does not exist: {folder}")
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            raise InsufficientCalibrationDataError(f"no images with extensions {IMAGE_EXTENSIONS} in {folder}")
        return self.calibrate_images(((p.name, load_gray_image(p)) for p in paths))

    def calibrate_images(self, images: Any) -> IntrinsicCalibrationResult:
        """``images``: iterable of (name, grayscale-or-BGR array or None). Images are processed lazily."""
        detections: list[tuple[str, np.ndarray]] = []
        rejected: list[ViewRecord] = []
        size: tuple[int, int] | None = None
        for name, img in images:
            if img is None:
                rejected.append(ViewRecord(name, False, "unreadable or empty image"))
                continue
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            this = (int(img.shape[1]), int(img.shape[0]))
            expected = self.config.expected_image_size or size
            if expected is not None and this != tuple(expected):
                why = "configured" if self.config.expected_image_size else "first usable image"
                rejected.append(ViewRecord(name, False, f"image size {this[0]}x{this[1]} differs from the {why} size "
                                                        f"{expected[0]}x{expected[1]}", this))
                continue
            corners = self.detect_corners(img)
            if corners is None:
                rejected.append(ViewRecord(name, False, f"chessboard with {self.config.pattern_size[0]}x"
                                                        f"{self.config.pattern_size[1]} inner corners not detected", this))
                continue
            size = size or this
            detections.append((name, corners))
        if size is None:
            raise InsufficientCalibrationDataError(
                f"no chessboard detected in any of {len(rejected)} image(s); check pattern_size counts INNER corners")
        return self.calibrate_from_detections(detections, size, rejected_views=rejected)

    # -- calibration ---------------------------------------------------------------------
    def calibrate_from_detections(self, detections: Sequence[tuple[str, np.ndarray]], image_size: tuple[int, int],
                                  rejected_views: Sequence[ViewRecord] = ()) -> IntrinsicCalibrationResult:
        cfg = self.config
        width, height = int(image_size[0]), int(image_size[1])
        if cfg.expected_image_size is not None and (width, height) != tuple(cfg.expected_image_size):
            raise InvalidCalibrationInputError(f"image size {width}x{height} differs from expected {cfg.expected_image_size}")
        n_points = cfg.pattern_size[0] * cfg.pattern_size[1]
        names, corner_sets = [], []
        for name, c in detections:
            arr = np.asarray(c, dtype=np.float64).reshape(-1, 2)
            if arr.shape[0] != n_points or not np.isfinite(arr).all():
                raise InvalidCalibrationInputError(f"{name}: expected {n_points} finite corners, got shape {arr.shape}")
            names.append(name); corner_sets.append(arr)
        if len(corner_sets) < cfg.min_views:
            raise InsufficientCalibrationDataError(
                f"{len(corner_sets)} usable view(s) found but min_views={cfg.min_views}; "
                f"rejected: {[(v.name, v.reason) for v in rejected_views]}")

        obj = chessboard_object_points(cfg.pattern_size, cfg.square_size_m)
        K, D, rvecs, tvecs = self._run_opencv(obj, corner_sets, (width, height))

        # Own reprojection error (authoritative; OpenCV's own figure is cross-checked in the tests).
        per_view = []
        for c, rv, tv in zip(corner_sets, rvecs, tvecs):
            proj = self._project(obj, rv, tv, K, D)
            per_view.append(float(np.sqrt(np.mean(np.sum((proj - c) ** 2, axis=1)))))
        total_sq = sum(float(np.sum((self._project(obj, rv, tv, K, D) - c) ** 2)) for c, rv, tv in zip(corner_sets, rvecs, tvecs))
        rms = math.sqrt(total_sq / (n_points * len(corner_sets)))

        quality, tilts = self._assess(K, D, rvecs, corner_sets, names, per_view, rms, (width, height))
        params = IntrinsicParameters(width, height, K, D, cfg.model, rms, tuple(per_view), len(corner_sets),
                                     cfg.pattern_size, cfg.square_size_m, quality)
        views = [ViewRecord(n, True, "", (width, height), e, t) for n, e, t in zip(names, per_view, tilts)] + list(rejected_views)
        result = IntrinsicCalibrationResult(params, quality, tuple(views), tuple(corner_sets), tuple(rvecs), tuple(tvecs))
        if quality.status == "poor" and cfg.enforce_quality:
            raise CalibrationQualityError("intrinsic calibration rejected: " + "; ".join(quality.issues), result)
        return result

    def _run_opencv(self, obj, corner_sets, size):
        cfg = self.config
        try:
            if cfg.model == "fisheye":
                objs = [obj.reshape(1, -1, 3) for _ in corner_sets]
                imgs = [c.reshape(1, -1, 2) for c in corner_sets]
                K = np.eye(3); D = np.zeros((4, 1))
                flags = _fisheye_flag("CALIB_RECOMPUTE_EXTRINSIC") | _fisheye_flag("CALIB_FIX_SKEW") | _fisheye_flag("CALIB_CHECK_COND")
                crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-12)
                _, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objs, imgs, size, K, D, None, None, flags, crit)
            else:
                flags = {"radial2": cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3, "standard5": 0,
                         "rational8": cv2.CALIB_RATIONAL_MODEL}[cfg.model]
                objs = [obj.astype(np.float32) for _ in corner_sets]
                imgs = [c.astype(np.float32).reshape(-1, 1, 2) for c in corner_sets]
                _, K, D, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, size, None, None, flags=flags)[:5]
        except cv2.error as exc:
            raise CalibrationQualityError(f"OpenCV could not solve the calibration (degenerate or ill-conditioned views): "
                                          f"{str(exc).splitlines()[0][:200]}") from None
        D = np.asarray(D, dtype=np.float64).reshape(-1)[:DISTORTION_MODELS[cfg.model]]
        if D.shape[0] < DISTORTION_MODELS[cfg.model]:
            D = np.concatenate([D, np.zeros(DISTORTION_MODELS[cfg.model] - D.shape[0])])
        K = np.asarray(K, dtype=np.float64)
        rvecs = [np.asarray(r, dtype=np.float64).reshape(3) for r in rvecs]
        tvecs = [np.asarray(t, dtype=np.float64).reshape(3) for t in tvecs]
        if not (np.isfinite(K).all() and np.isfinite(D).all() and all(np.isfinite(r).all() for r in rvecs)
                and all(np.isfinite(t).all() for t in tvecs)):
            raise CalibrationQualityError("calibration produced non-finite values")
        return K, D, rvecs, tvecs

    def _project(self, obj, rvec, tvec, K, D):
        if self.config.model == "fisheye":
            res, _ = cv2.fisheye.projectPoints(obj.reshape(1, -1, 3), rvec.reshape(3, 1), tvec.reshape(3, 1), K, D)
        else:
            res, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        return res.reshape(-1, 2)

    # -- quality -------------------------------------------------------------------------
    def _assess(self, K, D, rvecs, corner_sets, names, per_view, rms, size):
        cfg = self.config
        width, height = size
        issues_poor: list[str] = []
        issues_warn: list[str] = []
        metrics: dict[str, float] = {"rms_px": rms, "max_view_rms_px": max(per_view)}

        if rms > cfg.max_rms_px:
            issues_poor.append(f"overall RMS {rms:.3f} px exceeds max_rms_px={cfg.max_rms_px}")
        flagged = [n for n, e in zip(names, per_view) if e > cfg.max_view_rms_px]
        if flagged:
            issues_warn.append(f"{len(flagged)} view(s) exceed max_view_rms_px={cfg.max_view_rms_px}: {flagged}")

        normals = []
        tilts = []
        for rv in rvecs:
            R, _ = cv2.Rodrigues(rv)
            n = R[:, 2]
            normals.append(n)
            tilts.append(math.degrees(math.acos(min(1.0, abs(float(n[2]))))))
        spread = 0.0
        for i in range(len(normals)):
            for j in range(i + 1, len(normals)):
                spread = max(spread, math.degrees(math.acos(min(1.0, abs(float(normals[i] @ normals[j]))))))
        metrics["orientation_spread_deg"] = spread
        metrics["max_board_tilt_deg"] = max(tilts)
        if spread < cfg.min_orientation_spread_deg:
            issues_poor.append(f"board orientations too similar (max angle between views {spread:.1f} deg < "
                               f"{cfg.min_orientation_spread_deg}); tilt the board in different directions")

        cells = set()
        for c in corner_sets:
            cx = np.clip((c[:, 0] / width * 8).astype(int), 0, 7)
            cy = np.clip((c[:, 1] / height * 6).astype(int), 0, 5)
            cells.update(zip(cx.tolist(), cy.tolist()))
        coverage = len(cells) / 48.0
        metrics["coverage_fraction"] = coverage
        if coverage < cfg.min_coverage_fraction:
            issues_poor.append(f"detected corners cover only {coverage:.0%} of the image "
                               f"(< {cfg.min_coverage_fraction:.0%}); move the board across the whole frame, especially the edges")

        areas = [cv2.contourArea(cv2.convexHull(c.astype(np.float32))) / (width * height) for c in corner_sets]
        metrics["min_view_area_fraction"] = float(min(areas))
        small = [n for n, a in zip(names, areas) if a < cfg.min_view_area_fraction]
        if small:
            issues_warn.append(f"{len(small)} view(s) show the board very small (< {cfg.min_view_area_fraction:.0%} of the image): {small}")

        fx, fy = K[0, 0], K[1, 1]
        metrics["aspect_deviation"] = abs(fx / fy - 1)
        if metrics["aspect_deviation"] > cfg.max_aspect_deviation:
            issues_warn.append(f"fx/fy differs from 1 by {metrics['aspect_deviation']:.1%} (square pixels expected)")
        off = math.hypot(K[0, 2] - (width - 1) / 2, K[1, 2] - (height - 1) / 2) / math.hypot(width, height)
        metrics["principal_point_offset_fraction"] = off
        if off > 0.15:
            issues_warn.append(f"principal point is {off:.0%} of the image diagonal away from the centre")
        if cfg.pattern_size[0] % 2 == cfg.pattern_size[1] % 2:
            issues_warn.append("pattern has both inner-corner counts of the same parity: board orientation is ambiguous "
                               "and corner order may flip between views; prefer one even and one odd count")

        # Distortion must be invertible over the whole image, otherwise undistortion/BEV folds back.
        invert_err = self._invertibility_error(K, D, size)
        metrics["max_roundtrip_error_px"] = invert_err
        if not math.isfinite(invert_err) or invert_err > 0.5:
            issues_poor.append(f"distortion model is not invertible inside the image (round-trip error {invert_err:.2f} px): "
                               f"likely over-fitted or folding back; use a simpler model or more varied views")

        status = "poor" if issues_poor else ("warning" if issues_warn else "ok")
        quality = IntrinsicQuality(status, tuple(issues_poor + issues_warn), metrics, tuple(flagged))
        return quality, tilts

    def _invertibility_error(self, K, D, size) -> float:
        return lens_invertibility_error(IntrinsicParameters(size[0], size[1], K, D, self.config.model))
