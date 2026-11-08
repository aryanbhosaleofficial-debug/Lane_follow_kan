"""Independent review regressions: rasterized lines, physical circles and clutter."""
from __future__ import annotations

import cv2
import numpy as np

from .synthetic import scene


def edge_scene(slope: float, exit_row: int, mirror: bool = False):
    left = (0.0, slope, -slope * exit_row)
    right = (0.0, slope, 100 - slope * exit_row)
    image = scene((left, right))
    if mirror:
        image = image[:, ::-1].copy()
        left, right = tuple(-x for x in right), tuple(-x for x in left)
        left = (*left[:2], left[2] + 319)
        right = (*right[:2], right[2] + 319)
    return image, (left, right)


def arc_scene(radius: float, lane_width: float, orientation: float = 0.0,
              span: tuple[float, float] = (12, 228), size: tuple[int, int] = (320, 240)):
    """Concentric circles, exact normal width. u = cx-sqrt(r²-(v-cy)²).

    Radius is centreline radius, orientation is du/dv at v=120. No detector
    polynomial is used to build the ground truth or rasterization.
    """
    width, height = size
    cy = 120 - radius * orientation / np.sqrt(1 + orientation ** 2)
    cx = width / 2 + np.sqrt(radius ** 2 - (120 - cy) ** 2)
    rows = np.arange(span[0], span[1] + .1, .2)
    image = np.full((height, width, 3), 35, np.uint8)
    for r in (radius + lane_width / 2, radius - lane_width / 2):
        good = np.abs(rows - cy) < r
        u = cx - np.sqrt(r ** 2 - (rows[good] - cy) ** 2)
        points = np.rint(np.column_stack((u, rows[good]))).astype(np.int32)
        cv2.polylines(image, [points], False, (255, 255, 255), 5)
    return image, (float(cx), float(cy), radius, lane_width)


def circle_u(rows: np.ndarray, circle: tuple[float, float, float, float], radius_offset: float = 0) -> np.ndarray:
    cx, cy, radius, _ = circle
    return cx - np.sqrt((radius + radius_offset) ** 2 - (rows - cy) ** 2)


def clutter_scene(width: int, height: int) -> np.ndarray:
    image = scene(((0, 0, .22 * width), (0, 0, .63 * width)), width=width, height=height)
    rng = np.random.default_rng(20261108)
    for _ in range(160):
        u, v = rng.integers([5, 5], [width - 12, height - 18])
        cv2.line(image, (int(u), int(v)), (int(u + rng.integers(-3, 4)), int(v + rng.integers(4, 16))),
                 (230, 230, 230), 2)
    return image
