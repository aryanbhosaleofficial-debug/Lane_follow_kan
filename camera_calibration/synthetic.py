"""SYNTHETIC scene generation for tests and demonstrations.

Everything produced here is simulated geometry with *known* ground truth. Nothing in this module
describes a real camera, mounting pose, track or measurement, and results derived from it must
never be reported as real calibration evidence.

Conventions match the rest of M02: ground frame X forward, Y left, Z up (metres); camera frame is
OpenCV's (x right, y down, z forward).
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .intrinsics import IntrinsicParameters, chessboard_object_points

SYNTHETIC_NOTICE = "SYNTHETIC DATA - simulated geometry, not a real camera or track"


@dataclass(frozen=True)
class SyntheticCamera:
    """A virtual camera. ``dist`` follows the OpenCV ordering for ``model``."""
    width: int = 640
    height: int = 480
    fx: float = 520.0
    fy: float = 520.0
    cx: float = 319.5
    cy: float = 239.5
    dist: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    model: str = "standard5"

    def camera_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def intrinsics(self) -> IntrinsicParameters:
        return IntrinsicParameters(self.width, self.height, self.camera_matrix(), np.array(self.dist, float), self.model)


@dataclass(frozen=True)
class SyntheticPose:
    """Camera pose over the ground plane.

    ``pitch_down_deg`` tilts the optical axis below the horizon, ``yaw_deg`` turns it left
    (positive) about the ground Z axis, ``roll_deg`` rolls about the optical axis.
    """
    height_m: float = 0.25
    pitch_down_deg: float = 28.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0

    def ground_to_camera(self) -> tuple[np.ndarray, np.ndarray]:
        th, ps, ph = (math.radians(v) for v in (self.pitch_down_deg, self.yaw_deg, self.roll_deg))
        f = np.array([math.cos(ps) * math.cos(th), math.sin(ps) * math.cos(th), -math.sin(th)])
        r = np.array([math.sin(ps), -math.cos(ps), 0.0])
        d = np.cross(f, r)
        r, d = r * math.cos(ph) + d * math.sin(ph), d * math.cos(ph) - r * math.sin(ph)   # roll about f
        R = np.vstack([r, d, f])
        centre = np.array([self.x_m, self.y_m, self.height_m])
        return R, -R @ centre


def true_ground_homography(pose: SyntheticPose, undistorted_camera_matrix: np.ndarray) -> np.ndarray:
    """Exact ground (x, y, 1) -> undistorted-pixel homography of a synthetic pose, ||H||_F = 1, positive w."""
    R, t = pose.ground_to_camera()
    H = np.asarray(undistorted_camera_matrix, np.float64) @ np.column_stack([R[:, 0], R[:, 1], t])
    return H / np.linalg.norm(H)


# --------------------------------------------------------------------------------------------
# Projection and plane rendering
# --------------------------------------------------------------------------------------------
def project_camera_points(camera: SyntheticCamera, points_cam: np.ndarray) -> np.ndarray:
    """3-D points in the camera frame -> raw (distorted) pixels. Points behind the camera give NaN."""
    P = np.asarray(points_cam, np.float64).reshape(-1, 3)
    out = np.full((len(P), 2), np.nan)
    ok = P[:, 2] > 1e-9
    if ok.any():
        intr, zero = camera.intrinsics(), np.zeros(3)
        if camera.model == "fisheye":
            res, _ = cv2.fisheye.projectPoints(P[ok].reshape(1, -1, 3), zero, zero, intr.camera_matrix, intr.dist_coeffs)
        else:
            res, _ = cv2.projectPoints(P[ok], zero, zero, intr.camera_matrix, intr.dist_coeffs)
        out[ok] = res.reshape(-1, 2)
    return out


def project_ground_points(camera: SyntheticCamera, pose: SyntheticPose, ground_xy: Any) -> np.ndarray:
    """Ground (x, y) metres -> raw pixels (NaN if behind the camera)."""
    g = np.asarray(ground_xy, np.float64).reshape(-1, 2)
    R, t = pose.ground_to_camera()
    return project_camera_points(camera, np.column_stack([g, np.zeros(len(g))]) @ R.T + t)


@functools.lru_cache(maxsize=8)
def _pixel_rays(camera: SyntheticCamera) -> np.ndarray:
    us, vs = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
    raw = np.column_stack([us.ravel(), vs.ravel()]).astype(np.float64)
    norm = camera.intrinsics().undistort_points(raw, np.eye(3))
    rays = np.column_stack([norm, np.ones(len(norm))])
    rays.flags.writeable = False
    return rays


def render_plane_view(camera: SyntheticCamera, R: np.ndarray, t: np.ndarray, texture: np.ndarray, px_per_m: float,
                      origin_xy: tuple[float, float], background: int = 128, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Render what ``camera`` sees of the plane Z=0 of a frame whose pose is (R, t) = plane->camera.

    Plane point (X, Y) metres is looked up at texture pixel ((X-origin_x)*ppm, (Y-origin_y)*ppm).
    """
    rays = _pixel_rays(camera)
    n = R[:, 2]
    denom = rays @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (n @ t) / denom
    valid = np.isfinite(lam) & (lam > 0)
    lam = np.where(valid, lam, 0.0)
    plane = (rays * lam[:, None] - t) @ R
    mapx = ((plane[:, 0] - origin_xy[0]) * px_per_m).astype(np.float32)
    mapy = ((plane[:, 1] - origin_xy[1]) * px_per_m).astype(np.float32)
    mapx[~valid] = -1e4
    mapy[~valid] = -1e4
    shape = (camera.height, camera.width)
    return cv2.remap(texture, mapx.reshape(shape), mapy.reshape(shape), interpolation,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(background,) * 3)


# --------------------------------------------------------------------------------------------
# Chessboard
# --------------------------------------------------------------------------------------------
def chessboard_texture(pattern_size: tuple[int, int], square_m: float, px_per_square: int = 48) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Return (texture, px_per_m, origin_xy) for a board with ``pattern_size`` INNER corners.

    Board frame: inner corner (i, j) lies at (i*s, j*s, 0); X runs along the corner columns, Y
    along the rows. One white square of quiet zone surrounds the squares.
    """
    cols, rows = pattern_size
    ncx, ncy = cols + 3, rows + 3          # cells i in [-1, cols+1], j in [-1, rows+1]
    img = np.full((ncy * px_per_square, ncx * px_per_square, 3), 255, np.uint8)
    for j in range(-1, rows + 2):
        for i in range(-1, cols + 2):
            inside = 0 <= i <= cols and 0 <= j <= rows
            if inside and (i + j) % 2 == 0:
                y0, x0 = (j + 1) * px_per_square, (i + 1) * px_per_square
                img[y0:y0 + px_per_square, x0:x0 + px_per_square] = 0
    # texel centre of cell (-1) edge: X = -2s sits at texture coordinate -0.5
    return img, px_per_square / square_m, (-2 * square_m + 0.5 / (px_per_square / square_m), -2 * square_m + 0.5 / (px_per_square / square_m))


def render_chessboard(camera: SyntheticCamera, pattern_size: tuple[int, int], square_m: float, R_bc: np.ndarray, t_bc: np.ndarray,
                      noise_sigma: float = 0.8, seed: int = 0) -> np.ndarray:
    tex, ppm, origin = chessboard_texture(pattern_size, square_m)
    img = render_plane_view(camera, R_bc, t_bc, tex, ppm, origin, background=110)
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise_sigma, img.shape), 0, 255).astype(np.uint8)
    return img


def project_board_corners(camera: SyntheticCamera, pattern_size: tuple[int, int], square_m: float, R_bc: np.ndarray, t_bc: np.ndarray) -> np.ndarray:
    obj = chessboard_object_points(pattern_size, square_m)
    return project_camera_points(camera, obj @ R_bc.T + t_bc)


def random_board_poses(camera: SyntheticCamera, pattern_size: tuple[int, int], square_m: float, count: int, seed: int = 0,
                       max_tilt_deg: float = 50.0, depth_range: tuple[float, float] = (0.30, 0.75),
                       max_attempts: int = 20000) -> list[tuple[np.ndarray, np.ndarray]]:
    """Diverse board poses (board->camera) whose corners all project inside the image."""
    rng = np.random.default_rng(seed)
    K = camera.camera_matrix()
    cols, rows = pattern_size
    centre = np.array([(cols - 1) * square_m / 2, (rows - 1) * square_m / 2, 0.0])
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(max_attempts):
        if len(poses) == count:
            break
        u, v = rng.uniform(0.12, 0.88) * camera.width, rng.uniform(0.12, 0.88) * camera.height
        z = rng.uniform(*depth_range)
        pos = z * np.linalg.inv(K) @ np.array([u, v, 1.0])
        axis = rng.normal(size=3)
        axis[2] = 0.0
        axis /= np.linalg.norm(axis) + 1e-12
        tilt = math.radians(rng.uniform(0, max_tilt_deg))
        spin = math.radians(rng.uniform(-20, 20))
        R = cv2.Rodrigues(axis * tilt)[0] @ cv2.Rodrigues(np.array([0, 0, spin]))[0]
        t = pos - R @ centre
        corners = project_board_corners(camera, pattern_size, square_m, R, t)
        if not np.isfinite(corners).all():
            continue
        m = 12
        if corners[:, 0].min() < m or corners[:, 0].max() > camera.width - m or corners[:, 1].min() < m or corners[:, 1].max() > camera.height - m:
            continue
        if (R @ np.array([0, 0, 1.0]))[2] < 0.2:      # keep the printed side facing the camera
            continue
        poses.append((R, t))
    if len(poses) < count:
        raise RuntimeError(f"could only place {len(poses)} of {count} synthetic board poses")
    return poses


# --------------------------------------------------------------------------------------------
# Road scene with known marking geometry
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RoadScene:
    """A rendered-texture road plane plus the exact ground-truth marking geometry (metres)."""
    texture: np.ndarray
    px_per_m: float
    origin_xy: tuple[float, float]
    lines: tuple[tuple[tuple[float, float], tuple[float, float]], ...]      # straight marking segments
    arcs: tuple[tuple[float, float, float, float, float], ...]              # (cx, cy, radius, theta0, theta1) radians

    def render(self, camera: SyntheticCamera, pose: SyntheticPose, noise_sigma: float = 0.0, seed: int = 0) -> np.ndarray:
        R, t = pose.ground_to_camera()
        img = render_plane_view(camera, R, t, self.texture, self.px_per_m, self.origin_xy, background=30)
        if noise_sigma > 0:
            rng = np.random.default_rng(seed)
            img = np.clip(img.astype(np.float32) + rng.normal(0, noise_sigma, img.shape), 0, 255).astype(np.uint8)
        return img


def make_road_scene(kind: str = "straight", px_per_m: float = 500.0, marking_width_m: float = 0.02) -> RoadScene:
    """``straight``: two solid edge lines at y=+-0.25 and a dashed centre line (lane width 0.5 m is a
    SYNTHETIC value). ``curve``: the same lane turning left along a circular arc of radius 1.0 m
    about (1.2, 1.0)."""
    x0, x1, y0, y1 = -0.5, 4.0, -1.8, 1.8
    w, h = int((x1 - x0) * px_per_m), int((y1 - y0) * px_per_m)
    tex = np.full((h, w, 3), 30, np.uint8)
    lines: list = []
    arcs: list = []
    shift = 4

    def to_tex(p):
        return (int(round((p[0] - x0) * px_per_m * (1 << shift))), int(round((p[1] - y0) * px_per_m * (1 << shift))))

    def draw_line(a, b, width=marking_width_m):
        cv2.line(tex, to_tex(a), to_tex(b), (235, 235, 235), max(1, int(round(width * px_per_m))), cv2.LINE_AA, shift)

    def draw_arc(cx, cy, r, th0, th1, width=marking_width_m):
        ts = np.linspace(th0, th1, max(8, int(abs(th1 - th0) * r * px_per_m / 4)))
        pts = np.array([to_tex((cx + r * math.cos(t), cy + r * math.sin(t))) for t in ts], np.int32)
        cv2.polylines(tex, [pts.reshape(-1, 1, 2)], False, (235, 235, 235), max(1, int(round(width * px_per_m))), cv2.LINE_AA, shift)

    if kind == "straight":
        for y in (0.25, -0.25):
            a, b = (0.0, y), (3.6, y)
            lines.append((a, b)); draw_line(a, b)
        x = 0.1
        while x < 3.5:
            a, b = (x, 0.0), (x + 0.1, 0.0)
            lines.append((a, b)); draw_line(a, b, 0.015)
            x += 0.2
    elif kind == "curve":
        cx, cy = 1.2, 1.0
        for y, r in ((0.25, 0.75), (-0.25, 1.25)):
            a, b = (0.0, y), (cx, y)
            lines.append((a, b)); draw_line(a, b)
            arcs.append((cx, cy, r, -math.pi / 2, math.radians(15))); draw_arc(cx, cy, r, -math.pi / 2, math.radians(15))
    else:
        raise ValueError("kind must be 'straight' or 'curve'")
    return RoadScene(tex, px_per_m, (x0, y0), tuple(lines), tuple(arcs))
