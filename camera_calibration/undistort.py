"""Lens undistortion with cached remap tables.

Undistorted pixel coordinates are defined by ``UndistortionSpec.camera_matrix`` (``K_new``). Every
ground-plane homography in M02 maps into *these* coordinates, never into raw pixels, so a
homography is only valid together with the ``K_new`` it was estimated for.

Cropping is explicit: ``alpha=0`` keeps only valid pixels (the image is zoomed so no black border
remains, discarding some of the field of view), ``alpha=1`` keeps every source pixel (black
borders appear), ``alpha=None`` keeps ``K_new = K`` (no zoom; black borders appear for pincushion
distortion, whereas for barrel distortion every output pixel is sourced -- measured valid fraction 1.0 at k1 = -0.28 -- and
the outer part of the field of view falls outside the output). Output size always equals the input size.
"""
from __future__ import annotations

import functools
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ._util import array_to_list, frozen_array
from .errors import InvalidCalibrationInputError, InvalidFrameError, ResolutionMismatchError
from .intrinsics import IntrinsicParameters

INTERPOLATIONS = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}
# Source pixels needed on each side of a sample for a clean (border-free) interpolation result.
INTERPOLATION_MARGIN_PX = {"nearest": 0.0, "linear": 0.0, "cubic": 2.0}


def validate_pinhole_matrix(K: Any, name: str = "camera_matrix") -> np.ndarray:
    K = frozen_array(K, (3, 3), name)
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise InvalidCalibrationInputError(f"{name}: focal lengths must be positive")
    if not np.array_equal(K[2], [0.0, 0.0, 1.0]) or K[1, 0] != 0 or K[0, 1] != 0:
        raise InvalidCalibrationInputError(f"{name} must have the form [[fx,0,cx],[0,fy,cy],[0,0,1]]")
    return K


@dataclass(frozen=True, eq=False)
class UndistortionSpec:
    """Definition of the undistorted pixel coordinate system."""
    camera_matrix: np.ndarray                 # K_new
    alpha: float | None                       # free-scaling parameter used to derive K_new (None: K_new == K)
    output_size: tuple[int, int]              # (width, height)

    def __post_init__(self):
        object.__setattr__(self, "camera_matrix", validate_pinhole_matrix(self.camera_matrix, "undistorted camera_matrix"))
        if self.alpha is not None and not (isinstance(self.alpha, (int, float)) and not isinstance(self.alpha, bool)
                                           and 0.0 <= self.alpha <= 1.0):
            raise InvalidCalibrationInputError("alpha must be None or within [0, 1]")
        w, h = self.output_size
        if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
            raise InvalidCalibrationInputError("output_size must be positive integers")

    @classmethod
    def from_intrinsics(cls, intrinsics: IntrinsicParameters, alpha: float | None = 0.0) -> "UndistortionSpec":
        size = intrinsics.image_size
        if alpha is None:
            K_new = intrinsics.camera_matrix.copy()
        elif intrinsics.is_fisheye:
            K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                intrinsics.camera_matrix, intrinsics.dist_coeffs, size, np.eye(3), None, float(alpha), size, 1.0)
        else:
            K_new, _roi = cv2.getOptimalNewCameraMatrix(intrinsics.camera_matrix, intrinsics.dist_coeffs, size, float(alpha), size)
        K_new = np.asarray(K_new, dtype=np.float64)
        K_new[0, 1] = K_new[1, 0] = 0.0
        K_new[2] = [0.0, 0.0, 1.0]
        return cls(K_new, None if alpha is None else float(alpha), size)

    def scaled(self, width: int, height: int) -> "UndistortionSpec":
        """Spec for a uniformly rescaled image (pixel-centre convention); caller guarantees validity."""
        sx, sy = width / self.output_size[0], height / self.output_size[1]
        K = np.array(self.camera_matrix)
        K[0, 0] *= sx; K[1, 1] *= sy
        K[0, 2] = sx * (K[0, 2] + 0.5) - 0.5
        K[1, 2] = sy * (K[1, 2] + 0.5) - 0.5
        return UndistortionSpec(K, self.alpha, (width, height))

    def to_dict(self) -> dict:
        return {"camera_matrix": array_to_list(self.camera_matrix), "alpha": self.alpha, "output_size": list(self.output_size)}

    @classmethod
    def from_dict(cls, data: dict) -> "UndistortionSpec":
        if not isinstance(data, dict) or "camera_matrix" not in data or "output_size" not in data:
            raise InvalidCalibrationInputError("undistortion needs camera_matrix and output_size")
        size = data["output_size"]
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise InvalidCalibrationInputError("undistortion.output_size must be [width, height]")
        return cls(data["camera_matrix"], data.get("alpha"), (size[0], size[1]))


# -- map construction (cached; shared by Undistorter instances with identical parameters) -------
def _key(intr: IntrinsicParameters, spec: UndistortionSpec, fixed_point: bool) -> tuple:
    return (intr.model, intr.image_size, intr.camera_matrix.tobytes(), intr.dist_coeffs.tobytes(),
            spec.camera_matrix.tobytes(), spec.output_size, fixed_point)


@functools.lru_cache(maxsize=4)
def _build_maps(key: tuple) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    model, size, k_bytes, d_bytes, kn_bytes, out_size, fixed_point = key
    K = np.frombuffer(k_bytes, np.float64).reshape(3, 3).copy()
    D = np.frombuffer(d_bytes, np.float64).copy()
    Kn = np.frombuffer(kn_bytes, np.float64).reshape(3, 3).copy()
    if model == "fisheye":
        mx, my = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Kn, out_size, cv2.CV_32FC1)
    else:
        mx, my = cv2.initUndistortRectifyMap(K, D, np.eye(3), Kn, out_size, cv2.CV_32FC1)
    w, h = size
    valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1) & np.isfinite(mx) & np.isfinite(my)
    mask = np.where(valid, 255, 0).astype(np.uint8)
    mx = np.where(valid, mx, -1.0).astype(np.float32)
    my = np.where(valid, my, -1.0).astype(np.float32)
    if fixed_point:
        m1, m2 = cv2.convertMaps(mx, my, cv2.CV_16SC2)
    else:
        m1, m2 = mx, my
    for arr in (m1, m2, mask):
        if arr is not None:
            arr.flags.writeable = False
    return m1, m2, mask


class Undistorter:
    """Applies a precomputed undistortion remap. Thread-safe once constructed."""

    def __init__(self, intrinsics: IntrinsicParameters, spec: UndistortionSpec | None = None, *,
                 interpolation: str = "linear", fixed_point_maps: bool = False):
        if interpolation not in INTERPOLATIONS:
            raise InvalidCalibrationInputError(f"interpolation must be one of {sorted(INTERPOLATIONS)}")
        self.intrinsics = intrinsics
        self.spec = spec or UndistortionSpec.from_intrinsics(intrinsics)
        if self.spec.output_size != intrinsics.image_size:
            raise InvalidCalibrationInputError("undistortion output size must equal the calibrated image size")
        self.interpolation = interpolation
        self._flag = INTERPOLATIONS[interpolation]
        self._fixed = bool(fixed_point_maps)
        self._lock = threading.Lock()
        self._maps: tuple | None = None

    def _ensure_maps(self):
        maps = self._maps
        if maps is None:
            with self._lock:
                if self._maps is None:
                    self._maps = _build_maps(_key(self.intrinsics, self.spec, self._fixed))
                maps = self._maps
        return maps

    @property
    def valid_mask(self) -> np.ndarray:
        """uint8 (H, W) in undistorted coordinates: 255 where the source image provides data."""
        return self._ensure_maps()[2]

    @property
    def valid_fraction(self) -> float:
        return float(np.count_nonzero(self.valid_mask)) / self.valid_mask.size

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Return a new undistorted image; ``image`` (possibly read-only) is never modified.

        Pixels with no source data are 0; use ``valid_mask`` to tell them apart from black.
        """
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in (2, 3) \
                or (image.ndim == 3 and image.shape[2] not in (1, 3)) or image.size == 0:
            raise InvalidFrameError("undistort needs a non-empty uint8 image of shape HxW or HxWx3")
        h, w = image.shape[:2]
        if (w, h) != self.intrinsics.image_size:
            raise ResolutionMismatchError(f"image is {w}x{h} but the intrinsics were calibrated at "
                                          f"{self.intrinsics.image_width}x{self.intrinsics.image_height}")
        m1, m2, _ = self._ensure_maps()
        return cv2.remap(image, m1, m2, self._flag, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
