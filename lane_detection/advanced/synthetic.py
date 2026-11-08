"""Independent pixel-polynomial scenes. All geometry here is SYNTHETIC."""
from __future__ import annotations

from functools import lru_cache
import time

import cv2
import numpy as np

from camera_calibration import BEVConfig, BEVFrame, CalibrationProfile, CameraGeometry
from camera_calibration.profile import new_profile
from camera_calibration.plane_calibration import RoadPlaneCalibrator, RoadPlaneConfig
from camera_calibration.synthetic import SyntheticCamera, SyntheticPose, project_ground_points


@lru_cache(maxsize=4)
def synthetic_profile(width: int = 320, height: int = 240) -> CalibrationProfile:
    camera = SyntheticCamera()
    profile = new_profile(camera.intrinsics(), "SYNTHETIC M03 fixtures")
    x, y = np.meshgrid(np.linspace(.5, 1.6, 4), np.linspace(-.35, .35, 3))
    ground = np.column_stack((x.ravel(), y.ravel()))
    fit = RoadPlaneCalibrator(profile.intrinsics, profile.undistortion,
                              RoadPlaneConfig(origin_description="SYNTHETIC M03")).estimate(
                                  project_ground_points(camera, SyntheticPose(), ground), ground)
    # The frame's pixel truth is independent of this bookkeeping profile. Masks
    # below are synthetic full visibility, not claimed camera remap visibility.
    return profile.with_road_plane(fit).with_bev(BEVConfig(width, height, .004, width / 2, height - 1))


def scene(coefficients: tuple[tuple[float, float, float], ...] = ((0, 0, 70), (0, 0, 170)),
          *, width: int = 320, height: int = 240, dashed: bool = False, broken: bool = False,
          shadow: bool = False, noise: bool = False) -> np.ndarray:
    image = np.full((height, width, 3), 35, np.uint8)
    for index, abc in enumerate(coefficients):
        for start in range(6, height - 6, 60 if dashed else height):
            rows = np.arange(start, min(start + (32 if dashed else height), height - 6))
            points = np.rint(np.column_stack((np.polyval(abc, rows), rows))).astype(np.int32)
            cv2.polylines(image, [points], False, (255, 255, 255) if index % 2 == 0 else (0, 230, 230), 5)
    if broken:
        image[height // 2 - 18:height // 2 + 18] = 35
    if shadow:
        image[:height // 2] = (image[:height // 2].astype(float) * .48).astype(np.uint8)
    if noise:
        rng = np.random.default_rng(71)
        image = np.clip(image.astype(float) + rng.normal(0, 9, image.shape), 0, 255).astype(np.uint8)
        for _ in range(12):
            u, v = rng.integers([10, 10], [width - 10, height - 10])
            cv2.line(image, (int(u), int(v)), (int(u + 3), int(v + 3)), (220, 220, 220), 2)
    return image


def bev_fixture(image: np.ndarray, frame_id: int = 1, *, session: str = "SYNTHETIC-M03") -> BEVFrame:
    """Build an M02 contract fixture, NOT an M02 camera-remap integration test."""
    height, width = image.shape[:2]
    profile = synthetic_profile(width, height)
    mask = np.full((height, width), 255, np.uint8)
    mask.flags.writeable = False
    image = image.copy()
    image.flags.writeable = False
    return BEVFrame(image, mask, mask, CameraGeometry(profile), profile.bev, frame_id, frame_id,
                    session, "SYNTHETIC", time.time_ns(), time.monotonic_ns(), 640, 480,
                    time.monotonic_ns(), 0, profile.fingerprint())
