"""Road-plane homography estimation from measured ground / image correspondences.

The ground coordinates are physical measurements supplied by the caller (metres; X forward,
Y left; origin documented by the caller). Metric scale comes ONLY from those measurements --
no nominal lane width or other dimension is ever assumed.

What is checked, and what is not
--------------------------------
A homography that is merely invertible can still be physically wrong. This module therefore
rejects: too few / duplicate / collinear / non-finite points; denominators that change sign
(points straddling the horizon); ill-conditioned layouts; mirrored or axis-swapped coordinates
(the ground->image Jacobian must have the sign implied by ``camera_facing``); and a camera
that would sit below the road plane. It reports fit residuals, leave-one-out generalisation
error, and a pose decomposition (camera height / pitch / roll) that can be compared with a tape
measure. Thresholds are caller-configurable screening limits, not accuracy guarantees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ._util import as_points, is_positive_finite, is_real_number
from .errors import (DegenerateGeometryError, InsufficientCalibrationDataError, InvalidCalibrationInputError)
from .intrinsics import IntrinsicParameters
from .roadplane import CAMERA_FACING, W_EPS, RoadPlaneCalibration, apply_homography, convex_hull
from .undistort import UndistortionSpec

METHODS = ("least_squares", "ransac", "lmeds")
#: Heading of the optical axis (degrees from +X, positive toward +Y) expected for each declared facing.
EXPECTED_HEADING_DEG = {"forward": 0.0, "rear": 180.0}
HEADING_TOLERANCE_DEG = 45.0


@dataclass(frozen=True)
class RoadPlaneConfig:
    """Settings for road-plane homography estimation.

    Two orientation checks protect against mislabelled ground axes. (1) Handedness: with X
    forward, Y left and Z up, the ground->image mapping has a negative Jacobian determinant for
    *any* camera heading, so a positive one means the axes are mirrored or swapped (always
    enforced). (2) Heading: the recovered optical-axis direction must be within 45 degrees of
    ``camera_facing`` ("forward" = +X, "rear" = -X); a rotated axis labelling (e.g. X measured to
    the side) fails this. Use ``"unchecked"`` for deliberately side-looking cameras.
    ``max_condition_number`` bounds the conditioning of the Hartley-normalised homography: it
    measures how degenerate the point layout is, not physical accuracy.
    ``ransac_threshold_px`` is in undistorted pixels.
    """
    method: str = "least_squares"
    camera_facing: str = "forward"
    max_condition_number: float = 1e5
    ransac_threshold_px: float = 3.0
    min_hull_area_fraction: float = 0.01       # hull area / bounding-box area, in both image and ground space
    min_point_separation_px: float = 1.0
    min_point_separation_m: float = 1e-4
    orthonormality_warning: float = 0.10       # tolerance of the pose-consistency check (warning only)
    origin_description: str = ""

    def __post_init__(self):
        if self.method not in METHODS:
            raise InvalidCalibrationInputError(f"method must be one of {METHODS}")
        if self.camera_facing not in CAMERA_FACING:
            raise InvalidCalibrationInputError(f"camera_facing must be one of {CAMERA_FACING}")
        for name in ("max_condition_number", "ransac_threshold_px", "min_point_separation_px", "min_point_separation_m",
                     "orthonormality_warning"):
            if not is_positive_finite(getattr(self, name)):
                raise InvalidCalibrationInputError(f"{name} must be positive and finite")
        if not (is_real_number(self.min_hull_area_fraction) and 0 <= self.min_hull_area_fraction < 1):
            raise InvalidCalibrationInputError("min_hull_area_fraction must be within [0, 1)")
        if not isinstance(self.origin_description, str):
            raise InvalidCalibrationInputError("origin_description must be a string")


def _polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _hartley(points: np.ndarray) -> np.ndarray:
    c = points.mean(axis=0)
    s = math.sqrt(2.0) / max(float(np.mean(np.linalg.norm(points - c, axis=1))), 1e-300)
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def jacobian_determinants(H: np.ndarray, ground: np.ndarray) -> np.ndarray:
    """det d(u, v)/d(x, y) of the homography at each ground point; its sign is the mapping's handedness."""
    p = np.column_stack([ground, np.ones(len(ground))]) @ H.T
    w = p[:, 2]
    u, v = p[:, 0] / w, p[:, 1] / w
    a0, a1 = H[0, 0] - u * H[2, 0], H[0, 1] - u * H[2, 1]
    b0, b1 = H[1, 0] - v * H[2, 0], H[1, 1] - v * H[2, 1]
    return (a0 * b1 - a1 * b0) / (w * w)


def decompose_plane_pose(K_new: np.ndarray, H: np.ndarray) -> dict[str, float]:
    """Camera pose over the plane from ``H ~ K [r1 r2 t]``, as a plausibility cross-check.

    Returns the camera height above the plane, pitch below the horizon, roll, yaw (degrees; yaw
    positive toward +Y) and ``orthonormality_error``: how far the recovered r1, r2 are from unit
    length and mutual orthogonality. A large error means the homography is inconsistent with the
    intrinsics (wrong distortion model or K, bad points). A screening signal, not a measurement.
    """
    M = np.linalg.inv(K_new) @ H
    m1, m2, m3 = M[:, 0], M[:, 1], M[:, 2]
    lam = 0.5 * (np.linalg.norm(m1) + np.linalg.norm(m2))
    r1, r2, t = m1 / lam, m2 / lam, m3 / lam
    ortho = max(abs(float(r1 @ r2)), abs(float(np.linalg.norm(r1)) - 1.0), abs(float(np.linalg.norm(r2)) - 1.0))
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    centre = -R.T @ t
    optical, right = R[2, :], R[0, :]               # camera z / x axes expressed in ground coordinates
    pitch = math.degrees(math.asin(float(np.clip(-optical[2], -1, 1))))
    roll = math.degrees(math.asin(float(np.clip(-right[2] / max(math.cos(math.radians(pitch)), 1e-9), -1, 1))))
    yaw = math.degrees(math.atan2(optical[1], optical[0]))
    return {"camera_height_m": float(centre[2]), "camera_pitch_down_deg": pitch, "camera_roll_deg": roll,
            "camera_yaw_deg": yaw, "camera_x_m": float(centre[0]), "camera_y_m": float(centre[1]),
            "orthonormality_error": float(ortho)}


class RoadPlaneCalibrator:
    """Estimate the ground-plane homography from measured point correspondences."""

    def __init__(self, intrinsics: IntrinsicParameters, undistortion: UndistortionSpec, config: RoadPlaneConfig | None = None):
        if undistortion.output_size != intrinsics.image_size:
            raise InvalidCalibrationInputError("undistortion output size must equal the intrinsic image size")
        self.intrinsics = intrinsics
        self.undistortion = undistortion
        self.config = config or RoadPlaneConfig()

    # -- input validation ------------------------------------------------------------------
    def _prepare(self, image_points_px: Any, ground_points_m: Any, points_space: str):
        if points_space not in ("raw", "undistorted"):
            raise InvalidCalibrationInputError("points_space must be 'raw' or 'undistorted'")
        img = as_points(image_points_px, "image_points_px")
        ground = as_points(ground_points_m, "ground_points_m")
        if len(img) != len(ground):
            raise InvalidCalibrationInputError(f"{len(img)} image points but {len(ground)} ground points")
        raw = img if points_space == "raw" else None
        und = self.intrinsics.undistort_points(img, self.undistortion.camera_matrix) if raw is not None else img
        if not np.isfinite(und).all():
            raise InvalidCalibrationInputError("image points could not be undistorted (outside the invertible region of the lens model)")
        return raw, und, ground

    def _validate(self, ground: np.ndarray, und: np.ndarray, raw: np.ndarray | None) -> None:
        cfg = self.config
        n = len(ground)
        if n < 4:
            raise InsufficientCalibrationDataError(f"need at least 4 point correspondences, got {n}")
        w, h = self.intrinsics.image_size
        if raw is not None:
            slack = 0.5 * max(w, h)
            if (raw < -0.5 - slack).any() or (raw[:, 0] > w - 0.5 + slack).any() or (raw[:, 1] > h - 0.5 + slack).any():
                raise InvalidCalibrationInputError("some image points lie far outside the image; check the pixel coordinates")
        for name, pts, sep in (("image", und, cfg.min_point_separation_px), ("ground", ground, cfg.min_point_separation_m)):
            d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if d[i, j] < sep:
                raise DegenerateGeometryError(f"points {i} and {j} are (nearly) identical in {name} space (distance {d[i, j]:.3g}); "
                                              "repeated points are not allowed")
        self._check_spread("image", und)
        self._check_spread("ground", ground)

    def _check_spread(self, name: str, pts: np.ndarray) -> None:
        bbox = np.ptp(pts, axis=0)
        if bbox.min() <= 0 or bbox.min() < 1e-6 * bbox.max():
            raise DegenerateGeometryError(f"{name} points are collinear / lie on an axis-aligned line")
        sv = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
        if sv[1] < 1e-3 * sv[0]:
            raise DegenerateGeometryError(f"{name} points are (nearly) collinear; spread them over the road area")
        if len(pts) == 4:
            for tri in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                a, b, c = pts[list(tri)]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                if area < 1e-3 * bbox[0] * bbox[1]:
                    raise DegenerateGeometryError(f"three of the four {name} points are (nearly) collinear (points {tri})")
        hull = convex_hull(pts)
        frac = _polygon_area(hull) / (bbox[0] * bbox[1]) if len(hull) >= 3 else 0.0
        if frac < self.config.min_hull_area_fraction:
            raise DegenerateGeometryError(f"{name} points enclose only {frac:.2%} of their bounding box; they are almost collinear")

    # -- estimation ------------------------------------------------------------------------
    def estimate(self, image_points_px: Any, ground_points_m: Any, *, points_space: str = "raw") -> RoadPlaneCalibration:
        """Fit the homography.

        ``points_space="raw"``: image points were picked on the original (distorted) frame and are
        undistorted here with the intrinsics. ``"undistorted"``: they were picked on an image
        already undistorted with this profile's ``UndistortionSpec``; passing raw points as
        ``"undistorted"`` silently produces a wrong homography, so be explicit.
        """
        cfg = self.config
        raw, und, ground = self._prepare(image_points_px, ground_points_m, points_space)
        self._validate(ground, und, raw)
        n = len(ground)
        warnings: list[str] = []

        if cfg.method == "least_squares" or n == 4:
            if cfg.method != "least_squares":
                warnings.append(f"method {cfg.method!r} needs more than 4 points; used an exact 4-point fit")
            mask = np.ones(n, bool)
            method = "exact_4_point" if n == 4 else "least_squares"
        else:
            flag = cv2.RANSAC if cfg.method == "ransac" else cv2.LMEDS
            Hr, m = cv2.findHomography(ground, und, flag, cfg.ransac_threshold_px)
            if Hr is None or m is None:
                raise DegenerateGeometryError(f"{cfg.method} could not find a consistent homography")
            mask = m.reshape(-1).astype(bool)
            if mask.sum() < 4:
                raise DegenerateGeometryError(f"only {int(mask.sum())} inliers found; need at least 4")
            self._check_spread("ground (inliers)", ground[mask])
            self._check_spread("image (inliers)", und[mask])
            method = cfg.method
            if not mask.all():
                warnings.append(f"{int((~mask).sum())} point(s) rejected as outliers: indices {np.flatnonzero(~mask).tolist()}")
        exact = int(mask.sum()) == 4
        if exact:
            warnings.append("exactly 4 points: residuals are zero by construction and say nothing about accuracy; "
                            "check the result against extra measured points with validate()")

        H = self._fit(ground[mask], und[mask])
        H = self._canonicalise(H, ground[mask], und[mask])
        metrics = self._metrics(H, ground, und, mask, warnings)
        return RoadPlaneCalibration(H, self.undistortion.camera_matrix, self.intrinsics.image_size, ground, und, raw, mask, method,
                                    cfg.camera_facing, cfg.origin_description, convex_hull(ground[mask]), metrics, exact, tuple(warnings))

    @staticmethod
    def _fit(ground: np.ndarray, und: np.ndarray) -> np.ndarray:
        H, _ = cv2.findHomography(ground, und, 0)
        if H is None or not np.isfinite(H).all():
            raise DegenerateGeometryError("homography estimation failed (degenerate point configuration)")
        return H

    def _canonicalise(self, H: np.ndarray, ground: np.ndarray, und: np.ndarray) -> np.ndarray:
        """Unit norm, w > 0 over the calibration points; reject folded, mirrored or ill-conditioned fits."""
        cfg = self.config
        H = H / np.linalg.norm(H)
        w = np.column_stack([ground, np.ones(len(ground))]) @ H[2]
        if (w < 0).all():
            H, w = -H, -w
        if not (w > W_EPS).all():
            raise DegenerateGeometryError("calibration points are not all in front of the camera plane (denominators change sign): "
                                          "the points straddle the horizon, or the correspondences are inconsistent")
        cond = float(np.linalg.cond(_hartley(und) @ H @ np.linalg.inv(_hartley(ground))))
        if not math.isfinite(cond) or cond > cfg.max_condition_number:
            raise DegenerateGeometryError(f"homography is ill-conditioned (condition number {cond:.3g} > {cfg.max_condition_number:.3g}); the "
                                          "point layout is degenerate or the view is nearly edge-on")
        signs = np.sign(jacobian_determinants(H, ground))
        if not (signs < 0).all():
            if (signs > 0).all():
                raise DegenerateGeometryError(
                    "the ground axes look mirrored: with X forward, Y left and Z up the ground->image mapping must reverse orientation "
                    "(a right-handed ground frame seen from above). Check for swapped X/Y or a flipped sign in the measurements.")
            raise DegenerateGeometryError("the homography folds the ground plane (orientation differs between calibration points); "
                                          "check the point correspondences")
        return H

    def _metrics(self, H: np.ndarray, ground: np.ndarray, und: np.ndarray, mask: np.ndarray, warnings: list[str]) -> dict[str, float]:
        cfg = self.config
        g, u = ground[mask], und[mask]
        mapped, _ = apply_homography(H, g)
        err_px = np.linalg.norm(mapped - u, axis=1)
        back, back_ok = apply_homography(np.linalg.inv(H), u, require_positive_w=False)
        err_m = np.linalg.norm(back - g, axis=1)
        m: dict[str, float] = {
            "rms_reprojection_px": float(np.sqrt(np.mean(err_px ** 2))), "max_reprojection_px": float(err_px.max()),
            "rms_metric_error_m": float(np.sqrt(np.mean(err_m ** 2))), "max_metric_error_m": float(err_m.max()),
            "condition_number": float(np.linalg.cond(_hartley(u) @ H @ np.linalg.inv(_hartley(g)))),
        }
        # leave-one-out: honest generalisation error (only meaningful with a fit that has redundancy)
        if len(g) >= 5:
            loo_m, loo_px = [], []
            for i in range(len(g)):
                keep = np.arange(len(g)) != i
                try:
                    Hi = self._fit(g[keep], u[keep])
                    Hi = Hi / np.linalg.norm(Hi)
                    wi = np.column_stack([g[keep], np.ones(keep.sum())]) @ Hi[2]
                    if (wi < 0).all():
                        Hi = -Hi
                    pred, ok = apply_homography(Hi, g[i:i + 1])
                    back_i, ok2 = apply_homography(np.linalg.inv(Hi), u[i:i + 1], require_positive_w=False)
                except (DegenerateGeometryError, np.linalg.LinAlgError):
                    continue
                if ok.all() and ok2.all():
                    loo_px.append(float(np.linalg.norm(pred[0] - u[i])))
                    loo_m.append(float(np.linalg.norm(back_i[0] - g[i])))
            if loo_m:
                m["loo_rms_metric_error_m"] = float(np.sqrt(np.mean(np.square(loo_m))))
                m["loo_max_metric_error_m"] = float(max(loo_m))
                m["loo_rms_reprojection_px"] = float(np.sqrt(np.mean(np.square(loo_px))))
        pose = decompose_plane_pose(self.undistortion.camera_matrix, H)
        if pose["camera_height_m"] <= 0:
            raise DegenerateGeometryError(f"the fit places the camera {pose['camera_height_m']:.3g} m below the road plane; the coordinates "
                                          "are mirrored (check axis directions and signs) or the correspondences are wrong")
        if cfg.camera_facing != "unchecked":
            expected = EXPECTED_HEADING_DEG[cfg.camera_facing]
            off = abs((pose["camera_yaw_deg"] - expected + 180.0) % 360.0 - 180.0)
            if off > HEADING_TOLERANCE_DEG:
                raise DegenerateGeometryError(
                    f"the camera's optical axis points {pose['camera_yaw_deg']:.0f} deg from +X, but camera_facing={cfg.camera_facing!r} "
                    f"expects about {expected:.0f} deg (within {HEADING_TOLERANCE_DEG:.0f}). The measured X/Y axes may be rotated "
                    "(e.g. X measured sideways) or camera_facing is wrong; use 'unchecked' only for a deliberately side-looking camera.")
        m.update(pose)
        if pose["orthonormality_error"] > cfg.orthonormality_warning:
            warnings.append(f"the homography is not consistent with the intrinsics (pose orthonormality error {pose['orthonormality_error']:.2f} "
                            f"> {cfg.orthonormality_warning}): suspect intrinsics/distortion, point picking or non-planar points")
        return m

    # -- held-out validation ---------------------------------------------------------------
    def validate(self, calibration: RoadPlaneCalibration, image_points_px: Any, ground_points_m: Any, *,
                 points_space: str = "raw") -> dict[str, Any]:
        """Compare predicted and measured ground positions at points NOT used in the fit.

        This is the meaningful accuracy check for a 4-point fit. Returns per-point and summary
        metric errors (metres) plus whether any point fell outside the calibrated hull.
        """
        from .roadplane import points_in_convex_polygon
        raw, und, ground = self._prepare(image_points_px, ground_points_m, points_space)
        if (calibration.image_size != self.intrinsics.image_size
                or not np.allclose(calibration.undistorted_camera_matrix, self.undistortion.camera_matrix,
                                   rtol=1e-9, atol=1e-9)):
            raise InvalidCalibrationInputError("held-out validation uses a different camera resolution or undistortion matrix")
        if len(ground) == 0:
            raise InvalidCalibrationInputError("provide independent held-out points")
        if np.any(np.linalg.norm(ground[:, None, :] - calibration.ground_points_m[None, :, :], axis=2) < 1e-6):
            raise InvalidCalibrationInputError("held-out points repeat calibration points")
        pred, valid = apply_homography(np.linalg.inv(calibration.homography), und, require_positive_w=False)
        if not valid.all():
            raise DegenerateGeometryError("a held-out point projects at infinity; check correspondences and the calibration region")
        err = np.linalg.norm(pred - ground, axis=1)
        return {"errors_m": err.tolist(), "rms_m": float(np.sqrt(np.nanmean(err ** 2))), "max_m": float(np.nanmax(err)),
                "inside_calibrated_hull": points_in_convex_polygon(ground, calibration.calibrated_hull_m).tolist(),
                "predicted_ground_m": pred.tolist()}
