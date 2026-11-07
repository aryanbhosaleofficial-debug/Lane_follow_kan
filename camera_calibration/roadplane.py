"""Road-plane (ground) homography: calibration record and helpers.

The homography ``H`` maps homogeneous ground coordinates ``(x, y, 1)`` in metres (X forward,
Y left, origin documented by the user) to homogeneous *undistorted* pixel coordinates in the
``K_new`` system of :class:`camera_calibration.undistort.UndistortionSpec`. It is valid only for
points that lie ON the calibrated road plane.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._util import array_to_list, as_points, frozen_array, is_real_number
from .errors import ProfileError

GROUND_FRAME = "X forward, Y left, Z up; metres"
CAMERA_FACING = ("forward", "rear", "unchecked")
#: |w| below this (H is Frobenius-normalised) is treated as a point on the horizon / at infinity.
W_EPS = 1e-9


def apply_homography(H: np.ndarray, points: Any, *, require_positive_w: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``H`` to (N, 2) points. Returns (mapped (N, 2), valid (N,) bool).

    Rows are invalid (NaN) when the input is non-finite, the projective denominator is ~0, or --
    with ``require_positive_w`` -- negative, which means the point is behind the camera.
    """
    pts = as_points(points, allow_nonfinite=True)
    out = np.full_like(pts, np.nan)
    finite = np.isfinite(pts).all(axis=1)
    p = np.column_stack([np.where(np.isfinite(pts), pts, 0.0), np.ones(len(pts))]) @ np.asarray(H, np.float64).T
    w = p[:, 2]
    valid = finite & (w > W_EPS if require_positive_w else np.abs(w) > W_EPS)
    out[valid] = p[valid, :2] / w[valid, None]
    return out, valid


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull (Andrew's monotone chain) of (N, 2) points (viewed with X right, Y up)."""
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) <= 2:
        return np.array(pts, np.float64).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], np.float64)


def points_in_convex_polygon(points: Any, hull_ccw: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Boolean mask: points inside the CCW convex polygon grown outward by ``margin`` (same units).

    The margin is applied as a signed-distance test against each edge (exact for the polygon
    interior, a slightly rounded outline outside it).
    """
    p = as_points(points, allow_nonfinite=True)
    h = np.asarray(hull_ccw, np.float64)
    inside = np.isfinite(p).all(axis=1)
    if len(h) < 3:
        return np.zeros(len(p), bool)
    with np.errstate(invalid="ignore"):
        for a, b in zip(h, np.roll(h, -1, axis=0)):
            edge = b - a
            length = np.hypot(*edge)
            if length == 0:
                continue
            signed = (edge[0] * (p[:, 1] - a[1]) - edge[1] * (p[:, 0] - a[0])) / length   # >= 0 on the inner side (CCW)
            inside &= signed >= -margin
    return inside


@dataclass(frozen=True, eq=False)
class RoadPlaneCalibration:
    """Result of a road-plane calibration, including the evidence it was derived from."""
    homography: np.ndarray                         # ground (x, y, 1) -> undistorted pixels, ||H||_F = 1, positive w
    undistorted_camera_matrix: np.ndarray          # the K_new this homography is expressed in
    image_size: tuple[int, int]                    # (width, height) the calibration is valid for
    ground_points_m: np.ndarray                    # (N, 2) measured positions
    image_points_undistorted_px: np.ndarray        # (N, 2)
    image_points_raw_px: np.ndarray | None         # (N, 2) as originally picked, if supplied raw
    inlier_mask: np.ndarray                        # (N,) bool: points used in the final fit
    method: str
    camera_facing: str
    origin_description: str
    calibrated_hull_m: np.ndarray                  # (K, 2) CCW hull of the inlier ground points
    metrics: dict[str, float]
    exact_fit: bool                                # True when 4 points were used: residuals are zero by construction
    warnings: tuple[str, ...] = ()

    def __post_init__(self):
        H = frozen_array(self.homography, (3, 3), "homography")
        K = frozen_array(self.undistorted_camera_matrix, (3, 3), "undistorted_camera_matrix")
        ground = frozen_array(self.ground_points_m, None, "ground_points_m")
        und = frozen_array(self.image_points_undistorted_px, None, "image_points_undistorted_px")
        if ground.ndim != 2 or ground.shape[1] != 2 or und.shape != ground.shape or ground.shape[0] < 4:
            raise ProfileError("road plane needs >= 4 matching (N, 2) ground / image point sets")
        raw = None if self.image_points_raw_px is None else frozen_array(self.image_points_raw_px, ground.shape, "image_points_raw_px")
        mask = np.array(self.inlier_mask, dtype=bool).reshape(-1)
        if mask.shape != (ground.shape[0],) or mask.sum() < 4:
            raise ProfileError("inlier_mask must select at least 4 of the points")
        mask.flags.writeable = False
        hull = frozen_array(self.calibrated_hull_m, None, "calibrated_hull_m")
        if hull.ndim != 2 or hull.shape[1] != 2 or hull.shape[0] < 3:
            raise ProfileError("calibrated_hull_m must be a polygon with >= 3 vertices")
        w, h = self.image_size
        if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
            raise ProfileError("road plane image_size must be positive integers")
        if self.camera_facing not in CAMERA_FACING:
            raise ProfileError(f"camera_facing must be one of {CAMERA_FACING}")
        if not isinstance(self.origin_description, str):
            raise ProfileError("origin_description must be a string")
        clean = {}
        for k, v in self.metrics.items():
            if not (isinstance(k, str) and is_real_number(v) and math.isfinite(v)):
                raise ProfileError(f"road plane metric {k!r} must be a finite number")
            clean[k] = float(v)
        if abs(np.linalg.norm(H) - 1.0) > 1e-6:
            raise ProfileError("homography must be normalised to unit Frobenius norm")
        mapped, valid = apply_homography(H, ground[mask])
        if not valid.all():
            raise ProfileError("homography places calibration points at or behind the camera plane (w <= 0)")
        # Tamper / corruption check: the stored fit residual must be reproducible from the stored evidence.
        recomputed = float(np.max(np.linalg.norm(mapped - und[mask], axis=1)))
        stored = clean.get("max_reprojection_px")
        if stored is not None and abs(recomputed - stored) > 1e-6 + 1e-3 * max(recomputed, stored):
            raise ProfileError(f"stored max_reprojection_px {stored:.6g} does not match the homography and stored points "
                               f"({recomputed:.6g}): the profile was edited or is corrupt")
        object.__setattr__(self, "homography", H)
        object.__setattr__(self, "undistorted_camera_matrix", K)
        object.__setattr__(self, "ground_points_m", ground)
        object.__setattr__(self, "image_points_undistorted_px", und)
        object.__setattr__(self, "image_points_raw_px", raw)
        object.__setattr__(self, "inlier_mask", mask)
        object.__setattr__(self, "calibrated_hull_m", hull)
        object.__setattr__(self, "metrics", clean)
        object.__setattr__(self, "warnings", tuple(str(x) for x in self.warnings))

    def summary(self) -> str:
        m = self.metrics
        lines = [f"Road plane: {int(self.inlier_mask.sum())}/{len(self.inlier_mask)} points, method={self.method}, facing={self.camera_facing}",
                 f"  origin: {self.origin_description or '(not described)'}",
                 f"  frame: {GROUND_FRAME}"]
        for key in ("rms_reprojection_px", "max_reprojection_px", "rms_metric_error_m", "max_metric_error_m",
                    "loo_rms_metric_error_m", "condition_number", "camera_height_m", "camera_pitch_down_deg",
                    "camera_roll_deg", "camera_yaw_deg", "orthonormality_error"):
            if key in m:
                lines.append(f"  {key} = {m[key]:.5g}")
        if self.exact_fit:
            lines.append("  NOTE: exactly 4 points -> residuals are zero by construction; validate against held-out measurements")
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "frame": GROUND_FRAME, "homography": array_to_list(self.homography),
            "homography_maps": "ground (x, y, 1) metres -> undistorted pixels (in undistorted_camera_matrix coordinates)",
            "undistorted_camera_matrix": array_to_list(self.undistorted_camera_matrix), "image_size": list(self.image_size),
            "ground_points_m": array_to_list(self.ground_points_m),
            "image_points_undistorted_px": array_to_list(self.image_points_undistorted_px),
            "image_points_raw_px": None if self.image_points_raw_px is None else array_to_list(self.image_points_raw_px),
            "inlier_mask": [bool(v) for v in self.inlier_mask], "method": self.method, "camera_facing": self.camera_facing,
            "origin_description": self.origin_description, "calibrated_hull_m": array_to_list(self.calibrated_hull_m),
            "metrics": dict(self.metrics), "exact_fit": bool(self.exact_fit), "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RoadPlaneCalibration":
        if not isinstance(data, dict):
            raise ProfileError("road_plane must be an object")
        required = ("homography", "undistorted_camera_matrix", "image_size", "ground_points_m", "image_points_undistorted_px",
                    "inlier_mask", "method", "camera_facing", "calibrated_hull_m", "metrics", "exact_fit")
        missing = [k for k in required if k not in data]
        if missing:
            raise ProfileError(f"road_plane missing fields: {missing}")
        size = data["image_size"]
        if not isinstance(size, list) or len(size) != 2 or not all(isinstance(v, int) and not isinstance(v, bool) for v in size):
            raise ProfileError("road_plane.image_size must be [width, height] integers")
        if not isinstance(data["exact_fit"], bool) or not isinstance(data["metrics"], dict):
            raise ProfileError("road_plane.exact_fit must be a boolean and metrics an object")
        if not isinstance(data["inlier_mask"], list) or not all(isinstance(v, bool) for v in data["inlier_mask"]):
            raise ProfileError("road_plane.inlier_mask must be a list of booleans")
        return cls(data["homography"], data["undistorted_camera_matrix"], (size[0], size[1]), data["ground_points_m"],
                   data["image_points_undistorted_px"], data.get("image_points_raw_px"), np.array(data["inlier_mask"], bool),
                   str(data["method"]), data["camera_facing"], data.get("origin_description", ""), data["calibrated_hull_m"],
                   data["metrics"], data["exact_fit"], tuple(data.get("warnings", ())))
