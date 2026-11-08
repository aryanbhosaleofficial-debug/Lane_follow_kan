"""Immutable machine-readable M03 output contracts, independent of drawing/I/O."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Polynomial:
    """u(v) = a*v**2 + b*v + c, BEV pixels; valid only on v_min..v_max.

    Internal unsupported intervals are carried by Boundary, never hidden here.
    """
    coefficients: tuple[float, float, float]
    v_min: float
    v_max: float
    allow_extrapolation: bool = True

    def evaluate(self, v: Any) -> np.ndarray:
        rows = np.asarray(v, dtype=float)
        values = np.polyval(self.coefficients, rows)
        return values if self.allow_extrapolation else np.where((rows >= self.v_min) & (rows <= self.v_max), values, np.nan)

    def samples(self, count: int = 64) -> np.ndarray:
        v = np.linspace(self.v_min, self.v_max, count)
        return np.column_stack((self.evaluate(v), v))


@dataclass(frozen=True)
class FitDiagnostics:
    degree: int
    observed_points: int
    inlier_points: int
    residual_rms_px: float
    residual_p95_px: float
    coverage: float
    span_px: float
    near_coverage: float
    far_coverage: float
    max_gap_px: float
    max_curvature_px_inv: float
    condition_number: float
    hough_support_fraction: float = 0.0
    near_confidence: float = 0.0
    far_confidence: float = 0.0
    input_span_px: float = 0.0
    retained_fraction: float = 1.0
    signed_curvature_reference_px_inv: float = 0.0  # positive turning left as X increases
    endpoint_trimmed_points: int = 0
    observed_coverage_before_endpoint_trim: float = 0.0
    all_observation_rms_px: float = 0.0
    all_observation_p95_px: float = 0.0
    max_inlier_error_px: float = 0.0
    coherent_bias_max_px: float = 0.0
    marking_width_median_px: float = 0.0
    marking_width_p10_px: float = 0.0
    marking_width_p90_px: float = 0.0
    marking_width_ratio: float = 1.0
    residual_quality_factor: float = 1.0  # tolerance-normalized evidence, not probability


@dataclass(frozen=True)
class BEVMetadata:
    width_px: int
    height_px: int
    origin_u_px: float
    origin_v_px: float
    nominal_meters_per_pixel: float
    ground_origin_description: str


@dataclass(frozen=True)
class ExtractionDiagnostics:
    tracks_created: int = 0
    retired_tracks: int = 0
    discarded_short_tracks: int = 0
    peak_active_tracks: int = 0
    peak_completed_candidates: int = 0
    peak_stored_points: int = 0
    association_comparisons: int = 0
    budget_reason: str | None = None


@dataclass(frozen=True)
class Boundary:
    candidate_id: int
    polynomial: Polynomial
    observed: bool
    confidence: float
    reason_codes: tuple[str, ...]
    observed_intervals: tuple[tuple[float, float], ...]
    unsupported_intervals: tuple[tuple[float, float], ...]
    diagnostics: FitDiagnostics | None
    # Ground X forward, Y left. Omitted unless validated_geometry() succeeds.
    ground_points_m: tuple[tuple[float, float], ...] | None = None
    excluded_observation_intervals: tuple[tuple[float, float], ...] = ()
    limited_support: bool = False
    unsupported_far_intervals: tuple[tuple[float, float], ...] = ()
    model_misfit_intervals: tuple[tuple[float, float], ...] = ()
    marking_evidence_valid: bool = True


@dataclass(frozen=True)
class LanePair:
    left_id: int
    right_id: int
    centreline: Polynomial
    confidence: float
    inferred: bool
    width_mean_px: float
    width_variation: float
    width_mean_m: float | None = None
    centreline_points_px: tuple[tuple[float, float], ...] = ()
    # Each tuple is (left_u,left_v,right_u,right_v), at a normal correspondence.
    normal_correspondences_px: tuple[tuple[float, float, float, float], ...] = ()
    width_definition: str = "normal_distance"
    observed_fraction: float = 0.0
    inferred_fraction: float = 0.0
    shared_observed_fraction: float = 0.0
    shared_observed_span_px: float = 0.0
    unsupported_intervals: tuple[tuple[float, float], ...] = ()
    support_samples: int = 0
    reason_codes: tuple[str, ...] = ()
    limited_support: bool = False
    valid_support_v_px: tuple[float, float] | None = None
    observed_span_px: float = 0.0
    unsupported_far_intervals: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class StageTimings:
    preprocessing_ms: float = 0.0
    extraction_ms: float = 0.0
    fitting_ms: float = 0.0
    association_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True)
class LaneObservation:
    source_session_id: str
    source_frame_id: int
    source_id: str
    acquired_unix_ns: int
    acquired_monotonic_ns: int
    bev_sequence: int
    profile_fingerprint: str
    calibration_status: str
    metric_validated: bool
    validity: str
    confidence: float
    reason_codes: tuple[str, ...]
    boundaries: tuple[Boundary, ...]
    compatible_pairs: tuple[LanePair, ...]
    selected_pair: LanePair | None
    missing_boundaries: tuple[str, ...]
    timings: StageTimings
    coordinate_frame: str = "BEV: u right, v down, forward up"
    units: str = "pixels"
    brackets_origin: bool | None = None
    ego_offset_px: float | None = None   # lateral, positive vehicle-left of centre
    ego_offset_m: float | None = None    # same sign; validated scale only
    lane_reference_valid: bool = False
    ego_lane_valid: bool = False
    vehicle_reference_bev_px: tuple[float, float] | None = None
    reference_reason_codes: tuple[str, ...] = ()
    processing_frame_age_s: float = 0.0
    max_frame_age_s: float | None = None
    evidence_fraction: float = 0.0
    bev_metadata: BEVMetadata | None = None
    extraction_diagnostics: ExtractionDiagnostics | None = None
    limited_support: bool = False
    support_status: str = "none"
    usable_support_v_px: tuple[float, float] | None = None
    unsupported_far_intervals: tuple[tuple[float, float], ...] = ()
    reference_extrapolated: bool = False
    reference_extension_px: float = 0.0

    @property
    def valid(self) -> bool:
        return self.validity == "valid"

    def evaluate_supported_centreline(self, v: Any) -> np.ndarray:
        """Planner-facing query: NaN outside valid, jointly observed support.

        ``valid`` remains backward compatible and can describe a limited pair.
        This query never extrapolates, including for full-domain polynomials,
        and conservatively masks reported paint gaps rather than verifying them.
        A bounded vehicle-reference query does not enlarge this domain.
        """
        rows = np.asarray(v, dtype=float)
        pair = self.selected_pair
        if not self.valid or pair is None or self.usable_support_v_px is None:
            return np.full_like(rows, np.nan)
        lo, hi = self.usable_support_v_px
        lo, hi = max(lo, pair.centreline.v_min), min(hi, pair.centreline.v_max)
        supported = np.isfinite(rows) & (rows >= lo) & (rows <= hi)
        for start, end in pair.unsupported_intervals:
            supported &= ~((rows >= start) & (rows <= end))
        # Evaluate only supported rows: no extrapolated value is ever produced.
        values = np.full_like(rows, np.nan)
        values[supported] = pair.centreline.evaluate(rows[supported])
        return values

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe data; confidence is a heuristic evidence score."""
        return {**asdict(self), "valid": self.valid}


@dataclass(frozen=True, eq=False)
class DebugImages:
    """Read-only masks. Accepted pixels are a subset of raw marking evidence."""
    roi_mask: np.ndarray
    candidate_mask: np.ndarray
    accepted_mask: np.ndarray
    rejected_mask: np.ndarray
    edges: np.ndarray
    hough_segments: tuple[tuple[int, int, int, int], ...]
