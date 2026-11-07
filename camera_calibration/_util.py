"""Small shared helpers (validation and array handling)."""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from .errors import InvalidCalibrationInputError

UNIT_TO_METERS = {"m": 1.0, "cm": 0.01, "mm": 0.001}

# Iterative point undistortion criteria. OpenCV's default leaves ~1 px round-trip error under
# strong distortion (measured on OpenCV 5.0), so it is always overridden.
UNDISTORT_ITERATIONS = 100
UNDISTORT_EPSILON = 1e-12


_URL_USERINFO = re.compile(r"(?<=://)[^/@\s]*@")


def redact_credentials(text: str) -> str:
    """Replace URL userinfo (``user:pass@``) with ``***@``. Idempotent, and matches M01's own redaction."""
    return _URL_USERINFO.sub("***@", str(text))


def is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def is_positive_finite(value: Any) -> bool:
    if not is_real_number(value):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def is_positive_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value > 0


def as_points(points: Any, name: str = "points", *, allow_nonfinite: bool = False) -> np.ndarray:
    """Return an (N, 2) float64 copy. Accepts (2,) for a single point."""
    try:
        arr = np.array(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidCalibrationInputError(f"{name} must be numeric: {exc}") from None
    if arr.shape == (2,):
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise InvalidCalibrationInputError(f"{name} must have shape (N, 2), got {arr.shape}")
    if not allow_nonfinite and not np.isfinite(arr).all():
        raise InvalidCalibrationInputError(f"{name} contains non-finite values")
    return arr


def frozen_array(values: Any, shape: tuple[int, ...] | None = None, name: str = "array") -> np.ndarray:
    """Finite float64 read-only copy, optionally checking its shape."""
    try:
        arr = np.array(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidCalibrationInputError(f"{name} must be numeric: {exc}") from None
    if shape is not None and arr.shape != shape:
        raise InvalidCalibrationInputError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise InvalidCalibrationInputError(f"{name} contains non-finite values")
    arr.flags.writeable = False
    return arr


def array_to_list(arr: np.ndarray) -> list:
    return np.asarray(arr, dtype=np.float64).tolist()
