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
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ._util import as_points, is_positive_finite
from .errors import InvalidCalibrationInputError
from .intrinsics import lens_invertibility_error
from .mapper import CameraGeometry
from .profile import CalibrationProfile

_REPORT_KEY = secrets.token_bytes(32)  # process-local: reports must be issued by this validator

#: Evidence floor for ``ValidationReport.trusted_for`` (tune per project, then re-review). A report that passes
#: with one reference, or with a tolerance wider than the geometry it is meant to certify, proves nothing.
MIN_TRUSTED_REFERENCES = 3        # independent measured references (an aggregate "RMS error" row is not counted)
MAX_TRUSTED_TOLERANCE_M = 0.05    # every reference must have been judged against a tolerance no looser than this
_SYNTHETIC_MARKERS = ("synthetic", "simulat", "placeholder", "fake", "mock", "dummy")
_HULL_EPS_M = 1e-6                # measurements ON the calibrated hull boundary must not fail through float noise


@dataclass(frozen=True)
class TrustPolicy:
    """Project acceptance criteria; defaults are screening choices, not physical guarantees."""
    min_references: int = MIN_TRUSTED_REFERENCES
    max_tolerance_m: float = MAX_TRUSTED_TOLERANCE_M
    max_relative_error: float = 0.10
    min_span_fraction: float = 0.10

    def __post_init__(self):
        if isinstance(self.min_references, bool) or not isinstance(self.min_references, int) or self.min_references < 3:
            raise InvalidCalibrationInputError("min_references must be an integer >= 3")
        if not is_positive_finite(self.max_tolerance_m):
            raise InvalidCalibrationInputError("max_tolerance_m must be positive and finite")
        if not is_positive_finite(self.max_relative_error) or self.max_relative_error >= 1:
            raise InvalidCalibrationInputError("max_relative_error must be within (0, 1)")
        if not is_positive_finite(self.min_span_fraction) or self.min_span_fraction > 1:
            raise InvalidCalibrationInputError("min_span_fraction must be within (0, 1]")


def _looks_synthetic(origin_description: str) -> bool:
    """Heuristic guard, not proof: letters-only match so 'S Y N T H E T I C' or 'Synthetic-rig' are still caught."""
    return any(m in re.sub(r"[^a-z]", "", str(origin_description).lower()) for m in _SYNTHETIC_MARKERS)


@dataclass(frozen=True)
class MeasurementProvenance:
    """Operator attestation, not authentication of a physical measurement.

    The reference identifies the measurement log or marked target. A fresh runtime
    report must be generated from the measurements; a JSON summary cannot grant trust.
    """
    operator: str
    method: str
    reference: str
    independent_from_fit: bool

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.operator, self.method, self.reference)):
            raise InvalidCalibrationInputError("measurement provenance needs operator, method and reference")
        if self.independent_from_fit is not True:
            raise InvalidCalibrationInputError("independent_from_fit must be explicitly true")


def _report_signature(report: "ValidationReport") -> bytes:
    payload = repr((report.results, report.evidence, report.profile_fingerprint, report.provenance,
                    report.trust_policy, report.reference_scales, report.reference_points)).encode()
    return hmac.new(_REPORT_KEY, payload, hashlib.sha256).digest()


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
    provenance: MeasurementProvenance | None = None
    _signature: bytes = b""
    trust_policy: TrustPolicy = TrustPolicy()
    reference_scales: tuple[float, ...] = ()
    reference_points: tuple[tuple[float, float], ...] = ()

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.passed and math.isfinite(r.value) and math.isfinite(r.limit)
                                          for r in self.results)

    def trusted_for(self, profile: CalibrationProfile, policy: TrustPolicy | None = None) -> bool:
        """True only for a passing, attested report issued by this process's validator, bound to this exact profile,
        with distinct references, spatial coverage and tolerances meeting ``policy``.
        ``policy`` is the CONSUMER's acceptance rule (``BEVProcessor`` always supplies one); only when it is omitted does the
        check fall back to the policy the report's producer recorded, which proves nothing about what the consumer needs.
        This is a runtime integrity and evidence-floor check; it cannot prove a person really measured anything."""
        references = [r for r in self.results if r.name != "RMS error"]
        if not self.passed or profile.road_plane is None:
            return False
        if policy is not None and not isinstance(policy, TrustPolicy):
            raise InvalidCalibrationInputError("policy must be a TrustPolicy")
        policy = self.trust_policy if policy is None else policy
        points = np.asarray(self.reference_points, float).reshape(-1, 2)
        hull_span = np.ptp(profile.road_plane.calibrated_hull_m, axis=0)
        coverage = (len(points) >= 3 and np.isfinite(points).all()
                    and np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-6) == 2
                    and np.all(np.ptp(points, axis=0) >= hull_span * policy.min_span_fraction))
        return (self.evidence == "measured" and self.passed and self.provenance is not None
                and coverage and len(references) >= policy.min_references
                and len(self.reference_scales) == len(references)
                and all(r.unit == "m" and 0 < r.limit <= policy.max_tolerance_m
                        and r.limit <= scale * policy.max_relative_error
                        and r.value <= scale * policy.max_relative_error
                        for r, scale in zip(references, self.reference_scales))
                and self.profile_fingerprint == profile.fingerprint()
                and not _looks_synthetic(profile.road_plane.origin_description)
                and bool(self._signature) and hmac.compare_digest(self._signature, _report_signature(self)))

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
    def check_distances(self, measurements: Sequence[DistanceMeasurement], tolerance_m: float, *,
                        provenance: MeasurementProvenance | None = None,
                        trust_policy: TrustPolicy = TrustPolicy()) -> ValidationReport:
        """Compare distances measured in the BEV/ground plane against tape-measured reference lengths."""
        if not is_positive_finite(tolerance_m):
            raise InvalidCalibrationInputError("tolerance_m must be positive and finite")
        if not measurements:
            raise InvalidCalibrationInputError("provide at least one DistanceMeasurement")
        results, notes, reference_points, keys = [], [], [], set()
        for i, m in enumerate(measurements):
            pts = np.array([m.a_px, m.b_px], float)
            ground, valid = self.geometry.image_to_ground(pts, undistorted=m.undistorted, return_valid=True)
            key = tuple(sorted(map(tuple, np.round(ground, 6))))
            if key in keys:
                raise InvalidCalibrationInputError("duplicate distance reference, including reversed endpoints")
            keys.add(key)
            reference_points.extend(map(tuple, ground))
            label = m.label or f"distance {i}"
            if not valid.all():
                results.append(CheckResult(label, False, float("nan"), tolerance_m, "m", "a point has no ground position (at/above the horizon)"))
                continue
            d = float(np.linalg.norm(ground[0] - ground[1]))
            inside = self.geometry.in_calibrated_hull(ground, margin_m=_HULL_EPS_M)
            err = abs(d - m.distance_m)
            detail = f"measured {m.distance_m:.4f} m, computed {d:.4f} m, relative {err / m.distance_m:.2%}"
            fit_px = (self.profile.road_plane.image_points_undistorted_px if m.undistorted
                      else self.profile.road_plane.image_points_raw_px)
            if fit_px is None:
                fit_px = self.geometry.distort_points(self.profile.road_plane.image_points_undistorted_px)
            reused = (np.linalg.norm(pts[:, None, :] - fit_px[None, :, :], axis=2) < 0.5).any(axis=1).all()
            if reused:
                detail += " (both endpoints reuse calibration marks: circular evidence)"
            if not inside.all():
                detail += " (an endpoint lies outside the calibrated hull: extrapolation)"
            results.append(CheckResult(label, bool(not reused and inside.all() and err <= tolerance_m), err, tolerance_m, "m", detail))
        notes.append("only points on the road plane are valid; a raised mark (e.g. the top of a cone) is mapped to the ground behind it")
        return self._measurement_report(results, notes, provenance, trust_policy,
                                        tuple(m.distance_m for m in measurements), tuple(reference_points))

    def check_held_out_points(self, image_points_px: Any, ground_points_m: Any, tolerance_m: float, *,
                              undistorted: bool = False,
                              provenance: MeasurementProvenance | None = None,
                              trust_policy: TrustPolicy = TrustPolicy()) -> ValidationReport:
        """Predicted vs measured ground position at points that were NOT used for calibration."""
        if not is_positive_finite(tolerance_m):
            raise InvalidCalibrationInputError("tolerance_m must be positive and finite")
        px = as_points(image_points_px, "image_points_px")
        truth = as_points(ground_points_m, "ground_points_m")
        if len(px) != len(truth):
            raise InvalidCalibrationInputError("point counts differ")
        if len(np.unique(np.round(truth, 6), axis=0)) != len(truth):
            raise InvalidCalibrationInputError("duplicate held-out reference")
        fit = self.profile.road_plane.ground_points_m
        if np.any(np.linalg.norm(truth[:, None, :] - fit[None, :, :], axis=2) < 1e-6):
            raise InvalidCalibrationInputError("held-out ground points repeat calibration points; supply independent references")
        pred, valid = self.geometry.image_to_ground(px, undistorted=undistorted, return_valid=True)
        err = np.linalg.norm(pred - truth, axis=1)
        inside = self.geometry.in_calibrated_hull(truth, margin_m=_HULL_EPS_M)
        results = [CheckResult(f"point {i}", bool(valid[i] and inside[i] and err[i] <= tolerance_m), float(err[i]) if valid[i] else float("nan"),
                               tolerance_m, "m", "" if inside[i] else "outside calibrated hull") for i in range(len(px))]
        if valid.any():
            results.append(CheckResult("RMS error", bool(np.sqrt(np.nanmean(err[valid] ** 2)) <= tolerance_m),
                                       float(np.sqrt(np.nanmean(err[valid] ** 2))), tolerance_m, "m"))
        notes = ["points reused from the calibration set would be circular evidence; use independent measurements"]
        hull = self.profile.road_plane.calibrated_hull_m
        scale = float(np.max(np.linalg.norm(hull[:, None] - hull[None, :], axis=2)))
        return self._measurement_report(results, notes, provenance, trust_policy,
                                        (scale,) * len(truth), tuple(map(tuple, truth)))

    def _measurement_report(self, results, notes, provenance, policy, scales, points):
        if not isinstance(policy, TrustPolicy):
            raise InvalidCalibrationInputError("trust_policy must be TrustPolicy")
        if provenance is not None and not isinstance(provenance, MeasurementProvenance):
            raise InvalidCalibrationInputError("provenance must be MeasurementProvenance")
        report = ValidationReport(tuple(results), "measured", tuple(notes), self.profile.fingerprint(), provenance,
                                  trust_policy=policy, reference_scales=scales, reference_points=points)
        if provenance is not None:
            object.__setattr__(report, "_signature", _report_signature(report))
        return report
