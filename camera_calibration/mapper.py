"""Metric coordinate conversions and the fused raw-frame -> bird's-eye-view remap.

Coordinate systems handled here (all documented in ``bev.py`` and ``roadplane.py``):

* **raw image pixels**   - the camera frame as delivered by M01 (lens-distorted); OpenCV convention.
* **undistorted pixels** - ideal pinhole pixels in the ``K_new`` system of the profile's UndistortionSpec.
* **ground metres**      - X forward, Y left, on the calibrated road plane.
* **BEV pixels**         - the output grid (``BEVConfig``): u right, v down, forward is up.

Every conversion that crosses the image <-> ground boundary is only correct for points that lie ON
the calibrated road plane. A pixel belonging to a raised object (vehicle, sign, speed breaker)
maps to the ground point *behind* it along the viewing ray, not to its true position.
"""
from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ._util import as_points
from .bev import BEVConfig
from .errors import DegenerateGeometryError, InvalidBEVRegionError, InvalidFrameError, ProfileError, ResolutionMismatchError
from .profile import CalibrationProfile
from .roadplane import W_EPS, apply_homography, points_in_convex_polygon
from .undistort import INTERPOLATION_MARGIN_PX, INTERPOLATIONS, Undistorter

#: A BEV pixel / ground point is only accepted if distorting it and undistorting the result reproduces it to this
#: accuracy (ideal pixels). Valid points agree to ~1e-12 px; larger errors mean the lens polynomial has folded back
#: there, or is so close to folding that the iterative inverse no longer converges.
ROUNDTRIP_TOLERANCE_PX = 0.01
#: BEV pixels whose source footprint exceeds this many source pixels along an axis are undersampled.
UNDERSAMPLING_FOOTPRINT_PX = 2.0
_CHUNK_PIXELS = 200_000


class CameraGeometry:
    """All point conversions for one calibrated camera. Immutable and thread-safe."""

    def __init__(self, profile: CalibrationProfile, *, require_bev: bool = False):
        if profile.road_plane is None:
            raise ProfileError("profile has no road-plane calibration; metric conversions are unavailable")
        if require_bev and profile.bev is None:
            raise ProfileError("profile has no BEV grid configured")
        self.profile = profile
        self.intrinsics = profile.intrinsics
        self.road_plane = profile.road_plane
        self.bev = profile.bev
        self._K_new = profile.undistortion.camera_matrix
        self._K_inv = np.linalg.inv(self._K_new)
        self._H = profile.road_plane.homography
        self._H_inv = np.linalg.inv(self._H)

    # -- lens ------------------------------------------------------------------------------
    def undistort_points(self, raw_px: Any) -> np.ndarray:
        """Raw pixel coordinates -> undistorted pixel coordinates (``K_new`` system)."""
        return self.intrinsics.undistort_points(raw_px, self._K_new)

    def distort_points(self, undistorted_px: Any) -> np.ndarray:
        """Undistorted pixel coordinates -> raw pixel coordinates."""
        return self.intrinsics.distort_points(undistorted_px, self._K_new)

    # -- image <-> ground ----------------------------------------------------------------
    def image_to_ground(self, points_px: Any, *, undistorted: bool = False, on_invalid: str = "nan",
                        return_valid: bool = False):
        """Pixels -> ground (x forward, y left) metres, assuming each pixel shows a point ON the road plane.

        A pixel is invalid (NaN row) when its viewing ray does not meet the plane in front of the
        camera (at or above the horizon), when the lens model cannot undistort it, or when the
        input is non-finite. ``on_invalid="raise"`` raises instead of returning NaN.
        """
        self._check_mode(on_invalid)
        pts = as_points(points_px, "points_px", allow_nonfinite=True)
        und = pts if undistorted else self.undistort_points(pts)
        p = np.column_stack([np.where(np.isfinite(und), und, 0.0), np.ones(len(und))]) @ self._H_inv.T
        out = np.full((len(pts), 2), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = p[:, :2] / p[:, 2:3]
            forward_w = np.column_stack([g, np.ones(len(g))]) @ self._H[2]
        valid = np.isfinite(und).all(axis=1) & (np.abs(p[:, 2]) > 0) & np.isfinite(g).all(axis=1) & (forward_w > W_EPS)
        out[valid] = g[valid]
        return self._finish(out, valid, "image points", on_invalid, return_valid)

    def ground_to_image(self, ground_m: Any, *, undistorted: bool = False, on_invalid: str = "nan", return_valid: bool = False):
        """Ground metres -> pixels. Points at / behind the camera plane, or beyond where the lens model is invertible, are
        invalid (NaN).

        The result may lie outside the image; use :meth:`image_contains` to test that.
        """
        self._check_mode(on_invalid)
        g = as_points(ground_m, "ground_m", allow_nonfinite=True)
        und, valid = apply_homography(self._H, g)
        out = und if undistorted else self.distort_points(und)
        if not undistorted:
            # A lens polynomial can fold back far outside the field of view: a ground point out there would project to a
            # "ghost" pixel inside the image. Accept a raw pixel only if undistorting it returns the same ideal point.
            back = self.undistort_points(out)
            with np.errstate(invalid="ignore"):
                invertible = np.linalg.norm(back - und, axis=1) < ROUNDTRIP_TOLERANCE_PX
            valid = valid & np.isfinite(out).all(axis=1) & invertible
        return self._finish(out, valid, "ground points", on_invalid, return_valid)

    def image_contains(self, points_px: Any) -> np.ndarray:
        """Boolean mask: raw pixel coordinates inside the image area."""
        p = as_points(points_px, "points_px", allow_nonfinite=True)
        w, h = self.intrinsics.image_size
        with np.errstate(invalid="ignore"):
            return (p[:, 0] >= -0.5) & (p[:, 0] <= w - 0.5) & (p[:, 1] >= -0.5) & (p[:, 1] <= h - 0.5)

    # -- ground <-> BEV (exact affine) -----------------------------------------------------
    def ground_to_bev(self, ground_m: Any) -> np.ndarray:
        return self._require_bev().ground_to_bev(ground_m)

    def bev_to_ground(self, bev_px: Any) -> np.ndarray:
        return self._require_bev().bev_to_ground(bev_px)

    # -- composites ------------------------------------------------------------------------
    def image_to_bev(self, points_px: Any, *, undistorted: bool = False) -> np.ndarray:
        """Raw (or undistorted) pixels of ROAD points -> BEV pixel coordinates (NaN where invalid)."""
        return self.ground_to_bev(self.image_to_ground(points_px, undistorted=undistorted))

    def bev_to_image(self, bev_px: Any, *, undistorted: bool = False) -> np.ndarray:
        """BEV pixel coordinates -> raw (or undistorted) image pixels (NaN where behind the camera)."""
        return self.ground_to_image(self.bev_to_ground(bev_px), undistorted=undistorted)

    def ground_distance(self, a_m: Any, b_m: Any) -> np.ndarray:
        """Euclidean distance (metres) between corresponding ground points."""
        return np.linalg.norm(as_points(a_m, allow_nonfinite=True) - as_points(b_m, allow_nonfinite=True), axis=1)

    # -- region tests ----------------------------------------------------------------------
    def in_calibrated_hull(self, ground_m: Any, margin_m: float | None = None) -> np.ndarray:
        """Ground points inside the convex hull of the calibration points, grown by ``margin_m``
        (default: the BEV ``support_margin_m``, or 0 without a BEV). Outside it the homography extrapolates."""
        if margin_m is None:
            margin_m = self.bev.support_margin_m if self.bev is not None else 0.0
        return points_in_convex_polygon(ground_m, self.road_plane.calibrated_hull_m, margin_m)

    def check_ground_points(self, ground_m: Any) -> dict[str, np.ndarray]:
        """Where do these ground points stand?

        ``in_front``: ahead of the camera plane (homography denominator > 0). ``lens_valid``: additionally within the
        region where the lens model is invertible (so the projected pixel is meaningful). ``in_image``: projects inside
        the frame. ``in_calibrated_hull``: interpolation rather than extrapolation of the calibration. ``in_bev``:
        inside the BEV grid (only if one is configured).
        """
        g = as_points(ground_m, "ground_m", allow_nonfinite=True)
        _, in_front = apply_homography(self._H, g)
        px, lens_valid = self.ground_to_image(g, return_valid=True)
        result = {"in_front": in_front, "lens_valid": lens_valid, "in_image": lens_valid & self.image_contains(px),
                  "in_calibrated_hull": self.in_calibrated_hull(g)}
        if self.bev is not None:
            result["in_bev"] = self.bev.contains_bev(self.bev.ground_to_bev(g))
        return result

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _check_mode(on_invalid: str) -> None:
        if on_invalid not in ("nan", "raise"):
            raise ValueError("on_invalid must be 'nan' or 'raise'")

    @staticmethod
    def _finish(out: np.ndarray, valid: np.ndarray, what: str, on_invalid: str, return_valid: bool):
        out[~valid] = np.nan
        if on_invalid == "raise" and not valid.all():
            bad = np.flatnonzero(~valid)
            raise DegenerateGeometryError(f"{len(bad)} of {len(valid)} {what} have no valid mapping (at or above the horizon, behind the "
                                          f"camera, outside the lens model, or non-finite); first indices: {bad[:10].tolist()}")
        return (out, valid) if return_valid else out

    def _require_bev(self) -> BEVConfig:
        if self.bev is None:
            raise ProfileError("profile has no BEV grid configured")
        return self.bev


@dataclass(frozen=True)
class BEVMapInfo:
    """Properties of a built BEV remap (all computed from the maps, nothing assumed)."""
    valid_fraction: float                  # share of BEV pixels observed by the camera
    support_fraction: float                # share that is both observed and inside the calibrated hull (+ margin)
    footprint_px_min: float                # source pixels spanned by one BEV pixel (worst axis), over observed pixels
    footprint_px_median: float
    footprint_px_max: float
    fraction_undersampled: float           # observed pixels whose footprint exceeds UNDERSAMPLING_FOOTPRINT_PX
    build_seconds: float
    warnings: tuple[str, ...]


_BuiltMaps = tuple  # (map1, map2 | None, valid_mask, support_mask, info)
_cache_lock = threading.Lock()


def _compute_maps(profile: CalibrationProfile) -> _BuiltMaps:
    t0 = time.perf_counter()
    geo = CameraGeometry(profile, require_bev=True)
    bev = profile.bev
    assert bev is not None
    H, W = bev.shape
    w_src, h_src = profile.image_size
    margin = INTERPOLATION_MARGIN_PX[bev.interpolation]
    mapx = np.full((H, W), -1.0, np.float32)
    mapy = np.full((H, W), -1.0, np.float32)
    valid = np.zeros((H, W), bool)
    support = np.zeros((H, W), bool)
    K_inv = np.linalg.inv(profile.undistortion.camera_matrix)
    rows_per_chunk = max(1, _CHUNK_PIXELS // W)
    for r0 in range(0, H, rows_per_chunk):
        r1 = min(H, r0 + rows_per_chunk)
        us, vs = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(r0, r1, dtype=np.float64))
        uv = np.column_stack([us.ravel(), vs.ravel()])
        ground = bev.bev_to_ground(uv)
        und, front = apply_homography(geo._H, ground)
        norm = (np.column_stack([np.where(front[:, None], und, 0.0), np.ones(len(und))]) @ K_inv.T)[:, :2]
        raw = profile.intrinsics.project_normalized(np.where(front[:, None], norm, np.nan))
        back = profile.intrinsics.undistort_points(raw, profile.undistortion.camera_matrix)
        with np.errstate(invalid="ignore"):
            roundtrip_ok = np.linalg.norm(back - und, axis=1) < ROUNDTRIP_TOLERANCE_PX
            inside = ((raw[:, 0] >= margin) & (raw[:, 0] <= w_src - 1 - margin) & (raw[:, 1] >= margin) & (raw[:, 1] <= h_src - 1 - margin))
        ok = front & np.isfinite(raw).all(axis=1) & roundtrip_ok & inside
        shape = (r1 - r0, W)
        mapx[r0:r1] = np.where(ok, raw[:, 0], -1.0).astype(np.float32).reshape(shape)
        mapy[r0:r1] = np.where(ok, raw[:, 1], -1.0).astype(np.float32).reshape(shape)
        valid[r0:r1] = ok.reshape(shape)
        support[r0:r1] = (ok & points_in_convex_polygon(ground, profile.road_plane.calibrated_hull_m, bev.support_margin_m)).reshape(shape)

    n_valid = int(valid.sum())
    if n_valid == 0:
        raise InvalidBEVRegionError(
            f"no pixel of the requested BEV region ({bev.x_min_m:.2f}..{bev.x_max_m:.2f} m forward, {bev.y_min_m:.2f}..{bev.y_max_m:.2f} m "
            f"left) is observed by the camera. The region does not overlap the camera's view of the road plane; check the extents, the "
            f"origin, and that the profile matches the camera.")

    # source-pixel footprint of one BEV pixel from the map gradients (valid interior only)
    fx = np.where(valid, mapx, np.nan).astype(np.float64)
    fy = np.where(valid, mapy, np.nan).astype(np.float64)
    footprint = np.full((H, W), np.nan)
    if H > 1 and W > 1:
        dxu, dxv = np.gradient(fx, axis=1), np.gradient(fx, axis=0)
        dyu, dyv = np.gradient(fy, axis=1), np.gradient(fy, axis=0)
        footprint = np.maximum(np.hypot(dxu, dyu), np.hypot(dxv, dyv))
    fp = footprint[np.isfinite(footprint) & valid]
    warnings: list[str] = []
    valid_fraction = n_valid / valid.size
    support_fraction = int(support.sum()) / valid.size
    if valid_fraction < 0.25:
        warnings.append(f"only {valid_fraction:.0%} of the BEV region is observed by the camera")
    if support.sum() == 0:
        warnings.append("no observed BEV pixel lies inside the calibrated region: the whole BEV is extrapolated from the calibration")
    fraction_under = float((fp > UNDERSAMPLING_FOOTPRINT_PX).mean()) if fp.size else 0.0
    if fraction_under > 0.05:
        warnings.append(f"{fraction_under:.0%} of observed BEV pixels each cover more than {UNDERSAMPLING_FOOTPRINT_PX:.0f} source pixels "
                        "along an axis and no anti-aliasing filter is applied, so fine detail there can alias or shimmer: use a "
                        "smaller meters_per_pixel (a finer grid), or restrict the BEV to a farther range")
    info = BEVMapInfo(valid_fraction, support_fraction,
                      float(fp.min()) if fp.size else float("nan"), float(np.median(fp)) if fp.size else float("nan"),
                      float(fp.max()) if fp.size else float("nan"), fraction_under, 0.0, tuple(warnings))

    if bev.fixed_point_maps:
        m1, m2 = cv2.convertMaps(mapx, mapy, cv2.CV_16SC2, nninterpolation=(bev.interpolation == "nearest"))
    else:
        m1, m2 = mapx, mapy
    valid_mask = np.where(valid, 255, 0).astype(np.uint8)
    support_mask = np.where(support, 255, 0).astype(np.uint8)
    for arr in (m1, m2, valid_mask, support_mask):
        if arr is not None:
            arr.flags.writeable = False
    info = dataclasses.replace(info, build_seconds=time.perf_counter() - t0)
    return m1, m2, valid_mask, support_mask, info


_maps_cache: dict[str, _BuiltMaps] = {}
_MAPS_CACHE_LIMIT = 4


def _maps_for(profile: CalibrationProfile) -> _BuiltMaps:
    """Maps are pure functions of the geometry-relevant profile content, so identical profiles share them."""
    key = profile.fingerprint()
    with _cache_lock:
        hit = _maps_cache.get(key)
        if hit is not None:
            _maps_cache[key] = _maps_cache.pop(key)          # most-recently-used last
            return hit
    built = _compute_maps(profile)
    with _cache_lock:
        _maps_cache[key] = built
        while len(_maps_cache) > _MAPS_CACHE_LIMIT:
            _maps_cache.pop(next(iter(_maps_cache)))
    return built


def clear_map_cache() -> None:
    with _cache_lock:
        _maps_cache.clear()


class BEVMapper:
    """Warps raw camera frames into the metric bird's-eye view with ONE remap (undistortion fused with the homography).

    Built once per profile; ``warp`` is thread-safe and allocation-light. The returned image is
    only meaningful where ``valid_mask`` is 255 (the camera observed that ground point); other
    pixels hold ``border_value`` and must not be read as road.
    """

    def __init__(self, profile: CalibrationProfile):
        profile.require_metric()
        self.profile = profile
        self.profile_fingerprint = profile.fingerprint()       # computed once: hashing the profile per frame doubled process() cost
        self.geometry = CameraGeometry(profile, require_bev=True)
        self.bev: BEVConfig = profile.bev                      # type: ignore[assignment]
        self._flag = INTERPOLATIONS[self.bev.interpolation]
        self._m1, self._m2, self.valid_mask, self.support_mask, self.info = _maps_for(profile)
        self._border = (int(self.bev.border_value),) * 4

    @property
    def shape(self) -> tuple[int, int]:
        return self.bev.shape

    def warp(self, image: np.ndarray, dst: np.ndarray | None = None) -> np.ndarray:
        """Raw frame (HxWx3 or HxW uint8, possibly read-only) -> BEV image. ``image`` is never modified."""
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in (2, 3) \
                or (image.ndim == 3 and image.shape[2] not in (1, 3)) or image.size == 0:
            raise InvalidFrameError("warp needs a non-empty uint8 image of shape HxW or HxWx3")
        h, w = image.shape[:2]
        if (w, h) != self.profile.image_size:
            raise ResolutionMismatchError(f"image is {w}x{h} but the profile was calibrated at "
                                          f"{self.profile.image_size[0]}x{self.profile.image_size[1]}")
        if dst is not None:
            expected = self.bev.shape + image.shape[2:]
            if dst.shape != expected or dst.dtype != np.uint8 or not dst.flags.writeable or np.shares_memory(dst, image):
                raise InvalidFrameError(f"dst must be a writable, separate uint8 array of shape {expected}")
        return cv2.remap(image, self._m1, self._m2, self._flag, dst=dst, borderMode=cv2.BORDER_CONSTANT, borderValue=self._border)

    def undistorter(self, interpolation: str = "linear") -> Undistorter:
        """A matching :class:`Undistorter` (same ``K_new``) for diagnostics or consumers that want the undistorted frame."""
        return Undistorter(self.profile.intrinsics, self.profile.undistortion, interpolation=interpolation)
