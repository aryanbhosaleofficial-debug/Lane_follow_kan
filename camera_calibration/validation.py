"""Numerical validation of a calibration profile.

Two kinds of checks, deliberately kept apart:

* **Self-consistency** (``check_consistency``): properties that must hold for ANY correct
  implementation of the profile - conversions invert each other, the BEV grid agrees with the
  homography, the lens model is invertible over the image. These need no measurements and can
  be run on every profile. Passing them does NOT show the calibration matches the real world.
* **Measurement-based** (``check_distances``, ``check_held_out_points``): comparison against
  physical measurements YOU made on the real road plane (a tape measure, a ruler, a marker
  board of known size) that were NOT used to calibrate. This is the only evidence of real-world
  metric accuracy. The tolerances are yours to choose; nothing here invents one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ._util import as_points, is_positive_finite
from .errors import InvalidCalibrationInputError
from .intrinsics import lens_invertibility_error
from .mapper import CameraGeometry
from .profile import CalibrationProfile


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    value: float                  # the measured quantity (units in ``unit``)
    limit: float                  # the threshold it was compared against
    unit: str
    detail: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.value:.4g} {self.unit} (limit {self.limit:.4g}) {self.detail}".rstrip()


@dataclass(frozen=True)
class ValidationReport:
    results: tuple[CheckResult, ...]
    evidence: str                 # "self-consistency" or "measured"
    notes: tuple[str, ...] = ()
    profile_fingerprint: str | None = None

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def __str__(self) -> str:
        head = f"Validation ({self.evidence}): {'PASS' if self.passed else 'FAIL'}"
        return "\n".join([head] + [f"  {r}" for r in self.results] + [f"  note: {n}" for n in self.notes])


@dataclass(frozen=True)
class DistanceMeasurement:
    """A real, physically measured distance between two points visible in the image.

    ``a_px`` / ``b_px`` are pixel coordinates in the RAW frame (or undistorted frame if
    ``undistorted``); ``distance_m`` is what you measured with a tape measure. Both points must
    lie ON the road plane.
    """
    a_px: tuple[float, float]
    b_px: tuple[float, float]
    distance_m: float
    label: str = ""
    undistorted: bool = False

    def __post_init__(self):
        if not is_positive_finite(self.distance_m):
            raise InvalidCalibrationInputError("distance_m must be positive and finite")
        for p in (self.a_px, self.b_px):
            if len(p) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p):
                raise InvalidCalibrationInputError("pixel points must be two finite numbers")


class CalibrationValidator:
    """Runs validation checks against a metric profile."""

    def __init__(self, profile: CalibrationProfile):
        profile.require_metric()
        self.profile = profile
        self.geometry = CameraGeometry(profile, require_bev=True)

    # -- self-consistency ---------------------------------------------------------------
    def check_consistency(self, *, samples: int = 400, seed: int = 0, ground_tolerance_m: float = 1e-6,
                          pixel_tolerance_px: float = 1e-3) -> ValidationReport:
        """Round-trip checks on sample points drawn over the BEV region (only points the camera can see)."""
        bev = self.profile.bev
        geo = self.geometry
        rng = np.random.default_rng(seed)
        ground = np.column_stack([rng.uniform(bev.x_min_m, bev.x_max_m, samples), rng.uniform(bev.y_min_m, bev.y_max_m, samples)])
        px, front = geo.ground_to_image(ground, return_valid=True)
        seen = front & geo.image_contains(px)
        notes: list[str] = []
        results: list[CheckResult] = []
        if seen.sum() < 10:
            return ValidationReport((CheckResult("sample coverage", False, float(seen.sum()), 10.0, "points",
                                                 "fewer than 10 sample points of the BEV region are visible"),), "self-consistency")
        g, p = ground[seen], px[seen]
        back = geo.image_to_ground(p)
        results.append(CheckResult("ground->image->ground", bool(np.nanmax(np.linalg.norm(back - g, axis=1)) <= ground_tolerance_m),
                                   float(np.nanmax(np.linalg.norm(back - g, axis=1))), ground_tolerance_m, "m"))
        bev_px = geo.ground_to_bev(g)
        results.append(CheckResult("ground->BEV->ground", bool(np.abs(geo.bev_to_ground(bev_px) - g).max() <= 1e-9),
                                   float(np.abs(geo.bev_to_ground(bev_px) - g).max()), 1e-9, "m"))
        p_back = geo.bev_to_image(bev_px)
        err = float(np.nanmax(np.linalg.norm(p_back - p, axis=1)))
        results.append(CheckResult("image->BEV->image", err <= pixel_tolerance_px, err, pixel_tolerance_px, "px"))
        und = geo.undistort_points(p)
        rt = float(np.nanmax(np.linalg.norm(geo.distort_points(und) - p, axis=1)))
        results.append(CheckResult("lens distort/undistort round trip", rt <= pixel_tolerance_px, rt, pixel_tolerance_px, "px"))
        lens = lens_invertibility_error(self.profile.intrinsics)
        results.append(CheckResult("lens model invertible over the whole image", lens <= 0.5, lens, 0.5, "px",
                                   "" if lens <= 0.5 else "the distortion folds back or has no preimage inside the image: undistortion is wrong there"))
        # BEV scale: two points 1 m apart along each axis are exactly 1/mpp pixels apart
        step = 1.0 / bev.meters_per_pixel
        a = bev.ground_to_bev([[0.0, 0.0], [0.0, 0.0]])
        b = bev.ground_to_bev([[1.0, 0.0], [0.0, 1.0]])
        scale_err = float(max(abs(np.hypot(*(a[0] - b[0])) - step), abs(np.hypot(*(a[1] - b[1])) - step)))
        results.append(CheckResult("BEV scale (1 m in pixels)", scale_err < 1e-9, scale_err, 1e-9, "px"))
        mapper_valid = self._mapper_agreement(ground)
        results.append(mapper_valid)
        notes.append("self-consistency says the software is internally coherent; it says nothing about agreement with the real world")
        return ValidationReport(tuple(results), "self-consistency", tuple(notes))

    def _mapper_agreement(self, ground: np.ndarray) -> CheckResult:
        from .mapper import BEVMapper
        m = BEVMapper(self.profile)
        uv = self.profile.bev.ground_to_bev(ground)
        inside = self.profile.bev.contains_bev(uv)
        iu = np.clip(np.round(uv[inside, 0]).astype(int), 0, m.shape[1] - 1)
        iv = np.clip(np.round(uv[inside, 1]).astype(int), 0, m.shape[0] - 1)
        # a pixel the camera sees (image contains its projection) must not be flagged unobserved, except at the very edge
        px, front = self.geometry.ground_to_image(ground[inside], return_valid=True)
        w, h = self.profile.image_size
        interior = front & (px[:, 0] > 2) & (px[:, 0] < w - 3) & (px[:, 1] > 2) & (px[:, 1] < h - 3)
        flagged = int((m.valid_mask[iv, iu][interior] == 0).sum())
        return CheckResult("BEV valid mask vs projection", flagged == 0, float(flagged), 0.0, "pixels")

    # -- measurement-based --------------------------------------------------------------
    def check_distances(self, measurements: Sequence[DistanceMeasurement], tolerance_m: float) -> ValidationReport:
        """Compare distances measured in the BEV/ground plane against tape-measured reference lengths."""
        if not is_positive_finite(tolerance_m):
            raise InvalidCalibrationInputError("tolerance_m must be positive and finite")
        if not measurements:
            raise InvalidCalibrationInputError("provide at least one DistanceMeasurement")
        results, notes = [], []
        for i, m in enumerate(measurements):
            pts = np.array([m.a_px, m.b_px], float)
            ground, valid = self.geometry.image_to_ground(pts, undistorted=m.undistorted, return_valid=True)
            label = m.label or f"distance {i}"
            if not valid.all():
                results.append(CheckResult(label, False, float("nan"), tolerance_m, "m", "a point has no ground position (at/above the horizon)"))
                continue
            d = float(np.linalg.norm(ground[0] - ground[1]))
            inside = self.geometry.in_calibrated_hull(ground)
            err = abs(d - m.distance_m)
            detail = f"measured {m.distance_m:.4f} m, computed {d:.4f} m, relative {err / m.distance_m:.2%}"
            if not inside.all():
                detail += " (an endpoint lies outside the calibrated hull: extrapolation)"
            results.append(CheckResult(label, err <= tolerance_m, err, tolerance_m, "m", detail))
        notes.append("only points on the road plane are valid; a raised mark (e.g. the top of a cone) is mapped to the ground behind it")
        return ValidationReport(tuple(results), "measured", tuple(notes), self.profile.fingerprint())

    def check_held_out_points(self, image_points_px: Any, ground_points_m: Any, tolerance_m: float, *,
                              undistorted: bool = False) -> ValidationReport:
        """Predicted vs measured ground position at points that were NOT used for calibration."""
        if not is_positive_finite(tolerance_m):
            raise InvalidCalibrationInputError("tolerance_m must be positive and finite")
        px = as_points(image_points_px, "image_points_px")
        truth = as_points(ground_points_m, "ground_points_m")
        if len(px) != len(truth):
            raise InvalidCalibrationInputError("point counts differ")
        pred, valid = self.geometry.image_to_ground(px, undistorted=undistorted, return_valid=True)
        err = np.linalg.norm(pred - truth, axis=1)
        inside = self.geometry.in_calibrated_hull(truth)
        results = [CheckResult(f"point {i}", bool(valid[i] and err[i] <= tolerance_m), float(err[i]) if valid[i] else float("nan"),
                               tolerance_m, "m", "" if inside[i] else "outside calibrated hull") for i in range(len(px))]
        if valid.any():
            results.append(CheckResult("RMS error", bool(np.sqrt(np.nanmean(err[valid] ** 2)) <= tolerance_m),
                                       float(np.sqrt(np.nanmean(err[valid] ** 2))), tolerance_m, "m"))
        notes = ["points reused from the calibration set would be circular evidence; use independent measurements"]
        return ValidationReport(tuple(results), "measured", tuple(notes), self.profile.fingerprint())
