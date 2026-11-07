"""Bird's-eye-view grid definition, plus (further below) the remap builder and coordinate conversions.

Coordinate conventions (identical everywhere in M02)
----------------------------------------------------
Ground plane, metres:  X forward, Y left, Z up (right-handed).  The origin is a user-chosen,
documented reference point on the road plane (``RoadPlaneCalibration.origin_description``).

BEV image, pixels: OpenCV convention, u to the right and v downwards, with the *centre* of the
top-left pixel at (0, 0). The ground origin sits at BEV pixel (origin_u_px, origin_v_px)
(fractional values allowed; it may lie outside the image):

    u = origin_u_px - y / meters_per_pixel        (vehicle-left  -> image-left)
    v = origin_v_px - x / meters_per_pixel        (vehicle-forward -> image-up)

Hence forward is UP in the BEV image and the left of the vehicle is the LEFT of the image.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._util import as_points, is_positive_finite, is_positive_int, is_real_number
from .errors import InvalidBEVRegionError
from .undistort import INTERPOLATIONS

#: Guard against accidental multi-gigabyte BEV allocations.
MAX_BEV_PIXELS = 4_000_000


@dataclass(frozen=True)
class BEVConfig:
    """Output grid of the bird's-eye view.

    ``support_margin_m``: how far beyond the convex hull of the road-plane calibration points a
    BEV pixel may lie and still be counted as *supported* (interpolation rather than
    extrapolation of the calibration). It does not affect which pixels are observed.
    """
    width_px: int
    height_px: int
    meters_per_pixel: float
    origin_u_px: float
    origin_v_px: float
    interpolation: str = "linear"
    border_value: int = 0
    support_margin_m: float = 0.10
    fixed_point_maps: bool = False     # float32 maps measured ~3.7x faster than int16 on x86; re-measure on the target (see benchmark)

    def __post_init__(self):
        if not is_positive_int(self.width_px) or not is_positive_int(self.height_px):
            raise InvalidBEVRegionError("BEV width and height must be positive integers")
        object.__setattr__(self, "width_px", int(self.width_px))
        object.__setattr__(self, "height_px", int(self.height_px))
        if self.width_px * self.height_px > MAX_BEV_PIXELS:
            raise InvalidBEVRegionError(f"BEV of {self.width_px}x{self.height_px} exceeds MAX_BEV_PIXELS={MAX_BEV_PIXELS}")
        if not is_positive_finite(self.meters_per_pixel):
            raise InvalidBEVRegionError("meters_per_pixel must be positive and finite")
        for name in ("origin_u_px", "origin_v_px"):
            v = getattr(self, name)
            if not (is_real_number(v) and math.isfinite(v)):
                raise InvalidBEVRegionError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(v))
        if self.interpolation not in INTERPOLATIONS:
            raise InvalidBEVRegionError(f"interpolation must be one of {sorted(INTERPOLATIONS)}")
        bv = self.border_value
        if isinstance(bv, (bool, np.bool_)) or not isinstance(bv, (int, np.integer)) or not 0 <= bv <= 255:
            raise InvalidBEVRegionError("border_value must be an integer within 0..255")
        if not (is_real_number(self.support_margin_m) and math.isfinite(self.support_margin_m) and self.support_margin_m >= 0):
            raise InvalidBEVRegionError("support_margin_m must be finite and non-negative")

    # -- construction ----------------------------------------------------------------------
    @classmethod
    def from_extents(cls, x_min_m: float, x_max_m: float, y_min_m: float, y_max_m: float, meters_per_pixel: float,
                     **kwargs: Any) -> "BEVConfig":
        """Grid covering forward range [x_min, x_max] and lateral range [y_min (right), y_max (left)].

        The far/left edges (x_max, y_max) are exact; the near/right edges are rounded to a whole
        number of pixels, so read ``x_min_m`` / ``y_min_m`` from the result for the actual extents.
        """
        for name, v in (("x_min_m", x_min_m), ("x_max_m", x_max_m), ("y_min_m", y_min_m), ("y_max_m", y_max_m)):
            if not (is_real_number(v) and math.isfinite(v)):
                raise InvalidBEVRegionError(f"{name} must be finite")
        if not is_positive_finite(meters_per_pixel):
            raise InvalidBEVRegionError("meters_per_pixel must be positive and finite")
        if x_max_m <= x_min_m or y_max_m <= y_min_m:
            raise InvalidBEVRegionError("extents need x_max > x_min and y_max > y_min")
        width = round((y_max_m - y_min_m) / meters_per_pixel)
        height = round((x_max_m - x_min_m) / meters_per_pixel)
        if width < 1 or height < 1:
            raise InvalidBEVRegionError("extents are smaller than one BEV pixel")
        return cls(int(width), int(height), float(meters_per_pixel), y_max_m / meters_per_pixel - 0.5,
                   x_max_m / meters_per_pixel - 0.5, **kwargs)

    # -- derived extents (metres) ---------------------------------------------------------
    @property
    def x_max_m(self) -> float:
        """Forward edge of the image (top)."""
        return (self.origin_v_px + 0.5) * self.meters_per_pixel

    @property
    def x_min_m(self) -> float:
        """Nearest edge of the image (bottom)."""
        return (self.origin_v_px - (self.height_px - 0.5)) * self.meters_per_pixel

    @property
    def y_max_m(self) -> float:
        """Left edge of the image."""
        return (self.origin_u_px + 0.5) * self.meters_per_pixel

    @property
    def y_min_m(self) -> float:
        """Right edge of the image."""
        return (self.origin_u_px - (self.width_px - 0.5)) * self.meters_per_pixel

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of the BEV image array."""
        return (self.height_px, self.width_px)

    # -- exact affine conversions ----------------------------------------------------------
    def ground_to_bev(self, ground_xy_m: Any) -> np.ndarray:
        """Ground (x forward, y left) metres -> BEV pixel (u, v). NaN rows stay NaN."""
        g = as_points(ground_xy_m, "ground_xy_m", allow_nonfinite=True)
        out = np.empty_like(g)
        out[:, 0] = self.origin_u_px - g[:, 1] / self.meters_per_pixel
        out[:, 1] = self.origin_v_px - g[:, 0] / self.meters_per_pixel
        out[~np.isfinite(g).all(axis=1)] = np.nan      # never emit a half-valid coordinate
        return out

    def bev_to_ground(self, bev_uv_px: Any) -> np.ndarray:
        """BEV pixel (u, v) -> ground (x forward, y left) metres. NaN rows stay NaN."""
        p = as_points(bev_uv_px, "bev_uv_px", allow_nonfinite=True)
        out = np.empty_like(p)
        out[:, 0] = (self.origin_v_px - p[:, 1]) * self.meters_per_pixel
        out[:, 1] = (self.origin_u_px - p[:, 0]) * self.meters_per_pixel
        out[~np.isfinite(p).all(axis=1)] = np.nan
        return out

    def contains_bev(self, bev_uv_px: Any) -> np.ndarray:
        """Boolean mask: BEV pixel coordinates inside the image area."""
        p = as_points(bev_uv_px, "bev_uv_px", allow_nonfinite=True)
        with np.errstate(invalid="ignore"):
            return (p[:, 0] >= -0.5) & (p[:, 0] <= self.width_px - 0.5) & (p[:, 1] >= -0.5) & (p[:, 1] <= self.height_px - 0.5)

    def ground_to_bev_matrix(self) -> np.ndarray:
        """3x3 homogeneous matrix equivalent to :meth:`ground_to_bev` (acting on (x, y, 1))."""
        s = 1.0 / self.meters_per_pixel
        return np.array([[0.0, -s, self.origin_u_px], [-s, 0.0, self.origin_v_px], [0.0, 0.0, 1.0]])

    # -- serialisation ---------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"width_px": self.width_px, "height_px": self.height_px, "meters_per_pixel": self.meters_per_pixel,
                "origin_u_px": self.origin_u_px, "origin_v_px": self.origin_v_px, "interpolation": self.interpolation,
                "border_value": self.border_value, "support_margin_m": self.support_margin_m,
                "fixed_point_maps": self.fixed_point_maps,
                "convention": "u right, v down, pixel centres at integers; u = origin_u - y/mpp, v = origin_v - x/mpp"}

    @classmethod
    def from_dict(cls, data: dict) -> "BEVConfig":
        if not isinstance(data, dict):
            raise InvalidBEVRegionError("bev must be an object")
        missing = [k for k in ("width_px", "height_px", "meters_per_pixel", "origin_u_px", "origin_v_px") if k not in data]
        if missing:
            raise InvalidBEVRegionError(f"bev missing fields: {missing}")
        kw = {k: data[k] for k in ("interpolation", "border_value", "support_margin_m", "fixed_point_maps") if k in data}
        if "fixed_point_maps" in kw and not isinstance(kw["fixed_point_maps"], bool):
            raise InvalidBEVRegionError("fixed_point_maps must be a boolean")
        return cls(data["width_px"], data["height_px"], data["meters_per_pixel"], data["origin_u_px"], data["origin_v_px"], **kw)
