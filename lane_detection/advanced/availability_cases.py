"""Independent availability fixtures. Geometry and clutter seeds are explicit."""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import replace

from .synthetic import scene


def availability_frame(image: np.ndarray, scale: int = 1):
    """Equivalent synthetic grid; physical scale changes with raster resolution."""
    from camera_calibration import BEVConfig, CameraGeometry
    from .synthetic import bev_fixture
    frame = bev_fixture(image)
    grid = BEVConfig(image.shape[1], image.shape[0], .004 / scale, image.shape[1] / 2, image.shape[0] - 1)
    profile = frame.geometry.profile.with_bev(grid)
    return replace(frame, bev=grid, geometry=CameraGeometry(profile), profile_fingerprint=profile.fingerprint())


def clutter_case(kind: str, count: int, seed: int, scale: int = 1,
                 geometry: str = "straight") -> np.ndarray:
    coefficients = ((0, 0, 70), (0, 0, 200))
    if geometry == "curve":
        coefficients = ((.001, -.24, 84.4), (.001, -.24, 214.4))
    if geometry == "competing":
        coefficients = ((0, 0, 50), (0, 0, 150), (0, 0, 250))
    image = scene(coefficients, dashed=geometry == "dashed")
    rng = np.random.default_rng(seed)
    for _ in range(count):
        u, v = map(int, rng.integers([6, 8], [312, 230]))
        if kind == "fragments":
            du, dv = int(rng.integers(-4, 5)), int(rng.integers(3, 10))
            cv2.line(image, (u, v), (u + du, v + dv), (235, 235, 235), 2)
        elif kind == "specks":
            cv2.circle(image, (u, v), int(rng.integers(1, 3)), (235, 235, 235), -1)
        else:
            raise ValueError("kind must be fragments or specks")
    return image if scale == 1 else cv2.resize(image, (320 * scale, 240 * scale), interpolation=cv2.INTER_NEAREST)


def transition_case(radius: float = 300, transition_row: int = 120, *,
                    direction: int = 1, dashed: bool = False, arc_visible_px: int | None = None,
                    scale: int = 1, missing: bool = False) -> np.ndarray:
    """Straight boundaries join concentric arcs tangentially, width=130 base px.

    Near field is v>=transition_row. Far clipping is explicit, not extrapolated.
    Direction -1 mirrors the physical scene. Scale resizes the complete raster.
    """
    image = np.full((240, 320, 3), 35, np.uint8)
    v = np.arange(6, 234, .2)
    centre, lane_width = 135., 130.
    for i, r in enumerate((radius + lane_width / 2, radius - lane_width / 2)):
        if missing and i == 1:
            continue
        good = (v >= transition_row) | (transition_row - v < r)
        if arc_visible_px is not None:
            good &= v >= transition_row - arc_visible_px
        rows = v[good]
        delta = np.minimum(rows - transition_row, 0)
        u = centre + radius - np.sqrt(r ** 2 - delta ** 2)
        points = np.rint(np.column_stack((u, rows))).astype(np.int32)
        if dashed:
            mask = ((rows - 6) % 54) < 30
            cuts = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
            for start, end in zip(cuts[::2], cuts[1::2]):
                cv2.polylines(image, [points[start:end]], False, (255, 255, 255), 5)
        else:
            cv2.polylines(image, [points], False, (255, 255, 255), 5)
    if direction == -1:
        image = image[:, ::-1].copy()
    return image if scale == 1 else cv2.resize(image, (320 * scale, 240 * scale), interpolation=cv2.INTER_NEAREST)
