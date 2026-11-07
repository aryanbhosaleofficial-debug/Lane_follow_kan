"""M02 - Camera calibration and bird's-eye view.

Pipeline: M01 FramePacket -> (lens undistortion + road-plane homography, fused in one remap) -> BEVFrame.

Typical use::

    from camera_input import CameraConfig, CameraManager
    from camera_calibration import CalibrationProfile, BEVProcessor, BEVWorker

    profile = CalibrationProfile.load("my_camera_profile.json")     # produced by the calibration tools
    with CameraManager(CameraConfig(source=0, width=1280, height=720)) as camera, \\
            BEVWorker(BEVProcessor(profile), camera) as worker:
        bev = worker.wait_for_bev(after_sequence=0, timeout=1.0)
        ground_xy_m = bev.geometry.bev_to_ground([[175.0, 250.0]])

Coordinate conventions (everywhere): ground metres with X forward, Y left, Z up; BEV image pixels with u right, v down and
forward = up. See README.md. A bird's-eye view is valid for points ON the calibrated road plane only.
"""
from .bev import BEVConfig
from .errors import (CalibrationError, CalibrationQualityError, DegenerateGeometryError, FrameUnavailableError,
                     InsufficientCalibrationDataError, InvalidBEVRegionError, InvalidCalibrationInputError, InvalidFrameError,
                     ProfileError, ResolutionMismatchError, SourceMismatchError, StaleFrameError)
from .intrinsics import (IntrinsicCalibrationConfig, IntrinsicCalibrationResult, IntrinsicCalibrator, IntrinsicParameters,
                         IntrinsicQuality)
from .mapper import BEVMapInfo, BEVMapper, CameraGeometry
from .pipeline import BEVFrame, BEVProcessor, BEVWorker, ManagerFrameSource, ProcessorConfig
from .plane_calibration import RoadPlaneCalibrator, RoadPlaneConfig
from .profile import CalibrationProfile, new_profile
from .roadplane import RoadPlaneCalibration
from .undistort import Undistorter, UndistortionSpec
from .validation import CalibrationValidator, DistanceMeasurement, ValidationReport

__all__ = [
    "BEVConfig", "BEVFrame", "BEVMapInfo", "BEVMapper", "BEVProcessor", "BEVWorker", "CalibrationError", "CalibrationProfile",
    "CalibrationQualityError", "CalibrationValidator", "CameraGeometry", "DegenerateGeometryError", "DistanceMeasurement",
    "FrameUnavailableError", "InsufficientCalibrationDataError", "IntrinsicCalibrationConfig", "IntrinsicCalibrationResult",
    "IntrinsicCalibrator", "IntrinsicParameters", "IntrinsicQuality", "InvalidBEVRegionError", "InvalidCalibrationInputError",
    "InvalidFrameError", "ManagerFrameSource", "ProcessorConfig", "ProfileError", "ResolutionMismatchError", "RoadPlaneCalibration",
    "RoadPlaneCalibrator", "RoadPlaneConfig", "SourceMismatchError", "StaleFrameError", "Undistorter", "UndistortionSpec",
    "ValidationReport", "new_profile",
]
