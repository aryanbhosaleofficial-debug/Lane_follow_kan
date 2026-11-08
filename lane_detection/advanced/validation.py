"""Strict M02 boundary: raw camera images are never interpreted as metric BEV."""
from __future__ import annotations

import numpy as np

from camera_calibration import BEVConfig, BEVFrame, CalibrationStatus, CameraGeometry


def validate_frame(frame: BEVFrame) -> None:
    if not isinstance(frame, BEVFrame):
        raise TypeError("M03 requires an M02 BEVFrame; raw images use the separate legacy detector")
    if not isinstance(frame.bev, BEVConfig) or not isinstance(frame.geometry, CameraGeometry):
        raise ValueError("BEVFrame must carry M02 BEVConfig and CameraGeometry")
    image = frame.image
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or image.shape[:2] != frame.bev.shape or min(image.shape[:2]) < 16):
        raise ValueError("BEV image must be HxWx3 uint8 and match BEVConfig (minimum 16x16)")
    if frame.pixel_format != "BGR":
        raise ValueError("M03 requires BGR")
    for name in ("valid_mask", "support_mask"):
        mask = getattr(frame, name)
        if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.shape != frame.bev.shape:
            raise ValueError(f"{name} must be a uint8 mask matching BEVConfig")
        if np.any((mask != 0) & (mask != 255)):
            raise ValueError(f"{name} must contain only 0 and 255")
    if np.any((frame.support_mask != 0) & (frame.valid_mask == 0)):
        raise ValueError("support_mask must be a subset of valid_mask")
    if frame.geometry.bev != frame.bev or frame.geometry.profile.fingerprint() != frame.profile_fingerprint:
        raise ValueError("BEV geometry/grid/profile metadata conflict")
    if not isinstance(frame.calibration_status, CalibrationStatus):
        raise ValueError("invalid M02 calibration status")
    if frame.calibration_status in {CalibrationStatus.INVALID_OR_INCOMPATIBLE, CalibrationStatus.UNCALIBRATED}:
        raise ValueError("incompatible or uncalibrated M02 frame")
    for name in ("source_frame_id", "bev_sequence", "acquired_unix_ns", "acquired_monotonic_ns"):
        v = getattr(frame, name)
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not isinstance(frame.source_session_id, str) or not isinstance(frame.source_id, str):
        raise ValueError("source identifiers must be strings")
