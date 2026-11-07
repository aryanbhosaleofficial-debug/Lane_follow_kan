"""Numerical and visual diagnostics for judging a calibration with your own images.

All drawing functions return NEW images; inputs are never modified. They only visualise what a
profile/result already contains -- they do not add information, and a good-looking overlay is not
proof of accuracy (compare against measurements: see ``validation.py``).
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from ._util import as_points
from .errors import InvalidCalibrationInputError, InvalidFrameError
from .intrinsics import IntrinsicCalibrationResult
from .mapper import CameraGeometry
from .pipeline import BEVFrame
from .profile import CalibrationProfile

GREEN, RED, YELLOW, CYAN, MAGENTA, WHITE = (0, 200, 0), (0, 0, 255), (0, 220, 255), (255, 220, 0), (255, 0, 255), (255, 255, 255)


def row_straightness(corners: np.ndarray, pattern_size: tuple[int, int]) -> dict[str, float]:
    """How straight the board's corner rows are.

    ``corners``: (cols*rows, 2) in OpenCV detection order. Every row of ``cols`` corners should
    lie on a straight line in an undistorted image of a flat board. Returns the RMS and maximum
    perpendicular distance (pixels) of corners from their row's best-fit line, over all rows.
    This is a necessary check only: it is blind to distortion that keeps lines straight.
    """
    cols, rows = pattern_size
    pts = as_points(corners, "corners")
    if len(pts) != cols * rows:
        raise InvalidCalibrationInputError(f"expected {cols * rows} corners, got {len(pts)}")
    sq, worst = [], 0.0
    for r in range(rows):
        row = pts[r * cols:(r + 1) * cols]
        centred = row - row.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        dist = centred @ vt[1]                       # component perpendicular to the best-fit direction
        sq.extend((dist ** 2).tolist())
        worst = max(worst, float(np.abs(dist).max()))
    return {"rms_px": float(np.sqrt(np.mean(sq))), "max_px": worst}


# -- helpers ---------------------------------------------------------------------------------------
def _canvas(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in (2, 3) or image.size == 0:
        raise InvalidFrameError("diagnostics need a non-empty uint8 image")
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()


def _put_label(img: np.ndarray, text: str, org: tuple[float, float], color=WHITE, scale: float = 0.5) -> None:
    """Draw outlined text, clamped so that it stays fully inside the image (labels near an edge are shifted, not cut off)."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    h, w = img.shape[:2]
    if not (np.isfinite(org[0]) and np.isfinite(org[1])):
        return
    x = int(min(max(round(org[0]), 2), max(2, w - tw - 2)))
    y = int(min(max(round(org[1]), th + 2), max(th + 2, h - base - 2)))
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_if_finite(fn, img, pt, *args, **kwargs) -> bool:
    if not np.isfinite(pt).all() or abs(pt[0]) > 1e6 or abs(pt[1]) > 1e6:
        return False
    fn(img, (int(round(pt[0])), int(round(pt[1]))), *args, **kwargs)
    return True


# -- intrinsic calibration ------------------------------------------------------------------------
def draw_corner_overlay(image: np.ndarray, corners: np.ndarray, pattern_size: tuple[int, int], *, label_first: bool = True) -> np.ndarray:
    """Detected chessboard corners, joined in detection order. Corner 0 is marked: it must be the SAME physical corner in
    every image, or the calibration is meaningless (check this on a few images, especially rotated ones)."""
    out = _canvas(image)
    pts = as_points(corners, "corners")
    cols, rows = pattern_size
    if len(pts) != cols * rows:
        raise InvalidCalibrationInputError(f"expected {cols * rows} corners, got {len(pts)}")
    cv2.drawChessboardCorners(out, (cols, rows), pts.astype(np.float32).reshape(-1, 1, 2), True)
    if label_first:
        _draw_if_finite(cv2.circle, out, pts[0], 9, MAGENTA, 2, cv2.LINE_AA)
        _put_label(out, "corner 0", (pts[0, 0] + 12, pts[0, 1] - 8), MAGENTA)
    return out


def draw_reprojection_error(image: np.ndarray, detected: np.ndarray, projected: np.ndarray, *, scale: float = 20.0) -> np.ndarray:
    """Detected (green) vs model-projected (red) corners with the residual vector enlarged by ``scale`` so that
    sub-pixel errors are visible. Systematic patterns (all arrows pointing outward, say) indicate a model problem."""
    out = _canvas(image)
    a, b = as_points(detected, "detected"), as_points(projected, "projected")
    if a.shape != b.shape:
        raise InvalidCalibrationInputError("detected and projected corner counts differ")
    for p, q in zip(a, b):
        _draw_if_finite(cv2.circle, out, p, 3, GREEN, 1, cv2.LINE_AA)
        _draw_if_finite(cv2.circle, out, q, 2, RED, -1, cv2.LINE_AA)
        tip = p + (q - p) * scale
        if np.isfinite(tip).all() and np.isfinite(p).all():
            cv2.line(out, (int(round(p[0])), int(round(p[1]))), (int(round(tip[0])), int(round(tip[1]))), YELLOW, 1, cv2.LINE_AA)
    _put_label(out, f"residual x{scale:g}", (8, 18), YELLOW)
    return out


def reprojection_summary(result: IntrinsicCalibrationResult) -> str:
    """Per-image reprojection error table, worst first, with the overall RMS."""
    rows = sorted((v for v in result.views if v.used), key=lambda v: -(v.rms_px or 0))
    lines = [f"overall RMS {result.intrinsics.rms_reprojection_px:.4f} px over {len(rows)} view(s)",
             f"{'image':<32} {'RMS px':>8} {'board tilt deg':>15}"]
    lines += [f"{v.name:<32} {v.rms_px:>8.4f} {v.tilt_deg:>15.1f}" for v in rows]
    return "\n".join(lines)


def draw_undistortion_comparison(raw: np.ndarray, undistorted: np.ndarray, *, grid: int = 8) -> np.ndarray:
    """Raw and undistorted images side by side with the same straight reference grid: straight real-world lines
    (door frames, floor edges, board rows) should follow the grid after undistortion."""
    a, b = _canvas(raw), _canvas(undistorted)
    if a.shape != b.shape:
        raise InvalidFrameError("raw and undistorted images must have the same shape")
    for img in (a, b):
        h, w = img.shape[:2]
        for i in range(1, grid):
            cv2.line(img, (int(w * i / grid), 0), (int(w * i / grid), h), CYAN, 1, cv2.LINE_AA)
            cv2.line(img, (0, int(h * i / grid)), (w, int(h * i / grid)), CYAN, 1, cv2.LINE_AA)
    _put_label(a, "raw", (8, 18)); _put_label(b, "undistorted", (8, 18))
    return np.hstack([a, b])


# -- road plane ----------------------------------------------------------------------------------
def draw_reference_points(image: np.ndarray, profile: CalibrationProfile, *, show_residuals: bool = True) -> np.ndarray:
    """Calibration reference points on the RAW image: where they were picked (green) and where the calibration says their
    measured ground positions project (red). Outliers (not used in the fit) are drawn hollow. Residuals are exact pixel
    errors in the raw image, not magnified."""
    if profile.road_plane is None:
        raise InvalidCalibrationInputError("profile has no road-plane calibration")
    rp = profile.road_plane
    out = _canvas(image)
    geo = CameraGeometry(profile)
    picked = rp.image_points_raw_px if rp.image_points_raw_px is not None else geo.distort_points(rp.image_points_undistorted_px)
    predicted = geo.ground_to_image(rp.ground_points_m)
    for i, (p, q, used, g) in enumerate(zip(picked, predicted, rp.inlier_mask, rp.ground_points_m)):
        color = GREEN if used else (160, 160, 160)
        _draw_if_finite(cv2.circle, out, p, 6, color, 2 if used else 1, cv2.LINE_AA)
        if show_residuals and np.isfinite(q).all():
            _draw_if_finite(cv2.circle, out, q, 3, RED, -1, cv2.LINE_AA)
            cv2.line(out, (int(round(p[0])), int(round(p[1]))), (int(round(q[0])), int(round(q[1]))), RED, 1, cv2.LINE_AA)
        _put_label(out, f"{i}: ({g[0]:.2f}, {g[1]:.2f}) m", (p[0] + 8, p[1] - 6), color, 0.4)
    _put_label(out, "green: picked pixel   red: predicted from measured ground", (8, 18), WHITE, 0.45)
    _put_label(out, "labels: (x forward, y left) metres", (8, 36), WHITE, 0.45)
    return out


def draw_ground_grid_on_image(image: np.ndarray, profile: CalibrationProfile, *, spacing_m: float = 0.25,
                              x_range: tuple[float, float] | None = None, y_range: tuple[float, float] | None = None) -> np.ndarray:
    """Project a metric ground grid onto the RAW image. If the calibration is right, grid lines run along real road features
    and their spacing matches a tape measure laid on the road."""
    if not (isinstance(spacing_m, (int, float)) and math.isfinite(spacing_m) and spacing_m > 0):
        raise InvalidCalibrationInputError("spacing_m must be positive and finite")
    if profile.road_plane is None:
        raise InvalidCalibrationInputError("profile has no road-plane calibration")
    geo = CameraGeometry(profile)
    bev = profile.bev
    xr = x_range or ((bev.x_min_m, bev.x_max_m) if bev else (0.0, 3.0))
    yr = y_range or ((bev.y_min_m, bev.y_max_m) if bev else (-1.0, 1.0))
    out = _canvas(image)
    xs = np.arange(math.ceil(xr[0] / spacing_m) * spacing_m, xr[1] + 1e-9, spacing_m)
    ys = np.arange(math.ceil(yr[0] / spacing_m) * spacing_m, yr[1] + 1e-9, spacing_m)
    fine = np.linspace
    for x in xs:
        line = geo.ground_to_image(np.column_stack([np.full(60, x), fine(yr[0], yr[1], 60)]))
        _polyline(out, line, CYAN if abs(x) > 1e-9 else YELLOW)
        p = geo.ground_to_image([[x, yr[0]]])[0]
        if np.isfinite(p).all():
            _put_label(out, f"x={x:.2f}", (min(max(p[0], 2), out.shape[1] - 60), min(max(p[1], 12), out.shape[0] - 4)), CYAN, 0.4)
    for y in ys:
        _polyline(out, geo.ground_to_image(np.column_stack([fine(xr[0], xr[1], 60), np.full(60, y)])), CYAN if abs(y) > 1e-9 else YELLOW)
    return out


def _polyline(img: np.ndarray, pts: np.ndarray, color) -> None:
    h, w = img.shape[:2]
    ok = np.isfinite(pts).all(axis=1) & (np.abs(pts) < 1e5).all(axis=1)
    run: list = []
    for p, good in zip(pts, ok):
        if good:
            run.append(p)
        elif len(run) > 1:
            cv2.polylines(img, [np.round(np.array(run)).astype(np.int32).reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA); run = []
        else:
            run = []
    if len(run) > 1:
        cv2.polylines(img, [np.round(np.array(run)).astype(np.int32).reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)


# -- bird's-eye view -------------------------------------------------------------------------------
def draw_bev_grid(bev: np.ndarray | BEVFrame, config=None, *, spacing_m: float = 0.25, label: bool = True) -> np.ndarray:
    """Metric grid on a BEV image: lines every ``spacing_m`` metres, axes through the ground origin (x forward = up, y left = left).
    Measure a real distance on the road and compare it with the grid spacing."""
    if isinstance(bev, BEVFrame):
        image, config = bev.image, bev.bev
    else:
        image = bev
    if config is None:
        raise InvalidCalibrationInputError("pass the BEVConfig together with a plain image")
    if not (isinstance(spacing_m, (int, float)) and math.isfinite(spacing_m) and spacing_m > 0):
        raise InvalidCalibrationInputError("spacing_m must be positive and finite")
    out = _canvas(image)
    if out.shape[:2] != config.shape:
        raise InvalidFrameError(f"image {out.shape[:2]} does not match the BEV grid {config.shape}")
    xs = np.arange(math.ceil(config.x_min_m / spacing_m) * spacing_m, config.x_max_m + 1e-9, spacing_m)
    ys = np.arange(math.ceil(config.y_min_m / spacing_m) * spacing_m, config.y_max_m + 1e-9, spacing_m)
    for x in xs:
        v = config.ground_to_bev([[x, 0.0]])[0, 1]
        cv2.line(out, (0, int(round(v))), (out.shape[1] - 1, int(round(v))), YELLOW if abs(x) < 1e-9 else CYAN, 1, cv2.LINE_AA)
        if label:
            _put_label(out, f"{x:.2f} m", (3, v - 3), CYAN, 0.4)
    for y in ys:
        u = config.ground_to_bev([[0.0, y]])[0, 0]
        cv2.line(out, (int(round(u)), 0), (int(round(u)), out.shape[0] - 1), YELLOW if abs(y) < 1e-9 else CYAN, 1, cv2.LINE_AA)
        if label:
            _put_label(out, f"{y:.2f}", (u + 2, 12), CYAN, 0.4)
    o = config.ground_to_bev([[0.0, 0.0]])[0]
    if config.contains_bev([o])[0]:
        cv2.circle(out, (int(round(o[0])), int(round(o[1]))), 4, RED, -1, cv2.LINE_AA)
    return out


def draw_valid_region(frame: BEVFrame | None = None, *, bev_image: np.ndarray | None = None, valid_mask: np.ndarray | None = None,
                      support_mask: np.ndarray | None = None) -> np.ndarray:
    """BEV with the trust regions colour-coded: magenta = NOT observed by the camera (content is fill, never road); green tint =
    observed but outside the calibrated region (extrapolated: metric error grows); untinted = observed and supported."""
    if frame is not None:
        bev_image, valid_mask, support_mask = frame.image, frame.valid_mask, frame.support_mask
    if bev_image is None or valid_mask is None or support_mask is None:
        raise InvalidCalibrationInputError("pass a BEVFrame, or bev_image together with valid_mask and support_mask")
    out = _canvas(bev_image)
    if valid_mask.shape != out.shape[:2] or support_mask.shape != out.shape[:2]:
        raise InvalidFrameError("masks must match the BEV image size")
    unobserved = valid_mask == 0
    extrapolated = (valid_mask > 0) & (support_mask == 0)
    out[extrapolated] = (out[extrapolated] * 0.6 + np.array([0, 50, 0])).astype(np.uint8)
    out[unobserved] = (95, 0, 95)
    return out


def side_by_side(raw: np.ndarray, bev: np.ndarray, *, height: int | None = None) -> np.ndarray:
    """Raw frame next to its BEV, scaled to a common height for display only (never feed this back into processing)."""
    a, b = _canvas(raw), _canvas(bev)
    h = height or max(a.shape[0], b.shape[0])
    a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h), interpolation=cv2.INTER_AREA)
    return np.hstack([a, b])


def save_image(path: Any, image: np.ndarray) -> None:
    """Write an image, creating folders; raises instead of silently failing."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix or ".png", image)
    if not ok:
        raise OSError(f"could not encode image for {p}")
    buf.tofile(str(p))
