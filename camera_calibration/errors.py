"""Exception hierarchy for M02 camera calibration and bird's-eye view."""
from __future__ import annotations


class CalibrationError(Exception):
    """Base class for all M02 errors."""


class InvalidCalibrationInputError(CalibrationError, ValueError):
    """Calibration inputs (images, points, parameters) are malformed or unusable."""


class InsufficientCalibrationDataError(CalibrationError):
    """Too few usable views or point correspondences."""


class CalibrationQualityError(CalibrationError):
    """The calibration ran but its quality is not acceptable.

    ``result`` carries the computed (rejected) result so a developer can inspect it.
    """

    def __init__(self, message: str, result: object | None = None):
        super().__init__(message)
        self.result = result


class DegenerateGeometryError(CalibrationError):
    """Point configuration or homography is degenerate, mirrored, or physically implausible."""


class ProfileError(CalibrationError, ValueError):
    """A calibration profile file is malformed, incomplete, or inconsistent."""


class ResolutionMismatchError(CalibrationError):
    """Frame resolution differs from the resolution the profile was calibrated at."""


class SourceMismatchError(CalibrationError):
    """Frame source is not one the profile is bound to."""


class InvalidBEVRegionError(CalibrationError, ValueError):
    """Requested BEV region is empty, unobservable, or inconsistent with the calibration."""


class InvalidFrameError(CalibrationError):
    """Frame is not a valid HxWx3 uint8 BGR image or its metadata is inconsistent."""


class StaleFrameError(CalibrationError):
    """Frame is older than the configured maximum age."""


class FrameUnavailableError(CalibrationError):
    """No frame could be obtained from the source.

    ``reason`` is one of ``timeout`` (nothing arrived in time), ``no_frame`` (nothing published yet),
    ``camera_error`` (the camera is in a terminal error state) or ``stopped``.
    """

    def __init__(self, message: str, reason: str = "timeout"):
        super().__init__(message)
        self.reason = reason
