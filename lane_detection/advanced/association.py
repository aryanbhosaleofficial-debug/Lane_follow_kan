"""Conservative lane-pair geometry, without driving-lane or route selection."""
from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import numpy as np

from .config import LaneConfig
from .models import Boundary, LanePair, Polynomial
from .geometry import normal_geometry
from .observability import pair_observability


@dataclass(frozen=True)
class Association:
    pairs: tuple[LanePair, ...]
    ambiguous: bool
    reason_codes: tuple[str, ...]


def _extrema(coefficients: np.ndarray, lo: float, hi: float) -> np.ndarray:
    rows = [lo, hi]
    a, b, _ = coefficients
    if abs(a) > 1e-14 and lo < -b / (2 * a) < hi:
        rows.append(-b / (2 * a))
    return np.asarray(rows)


def _inside_roi(poly: Polynomial, rows: np.ndarray, roi: np.ndarray) -> bool:
    u = np.rint(poly.evaluate(rows)).astype(int)
    v = np.rint(rows).astype(int)
    h, w = roi.shape
    if np.any((u < 0) | (u >= w) | (v < 0) | (v >= h)):
        return False
    return bool(np.all(roi[v, u] != 0))


def associate(boundaries: tuple[Boundary, ...], roi: np.ndarray, config: LaneConfig,
              validated_mpp: float | None = None) -> Association:
    """Use exact quadratic/normal intersections, with explicit supported domains."""
    if (config.lane_width_m is not None or config.max_curvature_m_inv is not None) and validated_mpp is None:
        return Association((), False, ("metric_constraints_unavailable",))
    minimum, maximum = config.min_lane_width_px, config.max_lane_width_px
    if config.lane_width_m is not None:
        minimum, maximum = (v / validated_mpp for v in config.lane_width_m)
    pairs = []
    reasons: set[str] = set()
    if any(not b.marking_evidence_valid for b in boundaries):
        reasons.add("marking_width_inconsistent")
        boundaries = tuple(b for b in boundaries if b.marking_evidence_valid)
    crossing = False
    for first, second in itertools.combinations(boundaries, 2):
        lo = max(first.polynomial.v_min, second.polynomial.v_min)
        hi = min(first.polynomial.v_max, second.polynomial.v_max)
        if hi - lo < config.min_pair_overlap_px:
            reasons.add("insufficient_pair_overlap")
            if lo - hi > config.max_gap_px:
                reasons.add("support_gap_exceeds_limit")
            continue
        left, right = sorted((first, second), key=lambda b: float(b.polynomial.evaluate((lo + hi) / 2)))
        diff = np.subtract(right.polynomial.coefficients, left.polynomial.coefficients)
        widths_extreme = np.polyval(diff, _extrema(diff, lo, hi))
        if np.min(widths_extreme) <= 0:
            crossing = True
            reasons.add("boundary_intersection")
            continue
        # Row extraction stops before a merge when paint becomes one broad run.
        # Two supported curves converging to paint-width separation are still a
        # fork/merge, even if their fitted domains end just before intersection.
        if (np.min(widths_extreme) <= config.max_marking_width_px
                and np.ptp(widths_extreme) > 2 * config.association_margin_px):
            crossing = True
            reasons.add("branch_geometry")
            continue
        # Rejected narrow pairs are evidence of competing paint, not evidence
        # that the remaining marking is the true edge. Keep that ambiguity even
        # when only one full-width adjacent pair survives geometric screening.
        if len(boundaries) > 2 and np.min(widths_extreme) <= minimum + 1e-6:
            crossing = True
            reasons.add("competing_close_markings")
        # A boundary in between makes this a multi-lane span, not an adjacent pair.
        rows_check = np.linspace(lo, hi, config.normal_samples)
        ul, ur = left.polynomial.evaluate(rows_check), right.polynomial.evaluate(rows_check)
        if any(b.candidate_id not in {left.candidate_id, right.candidate_id}
               and np.any((rows_check >= b.polynomial.v_min) & (rows_check <= b.polynomial.v_max)
                          & (b.polynomial.evaluate(rows_check) > ul) & (b.polynomial.evaluate(rows_check) < ur))
               for b in boundaries):
            reasons.add("intervening_marking")
            continue
        geometry = normal_geometry(left.polynomial, right.polynomial, config)
        widths = geometry.widths
        if len(widths) and (np.min(widths) < minimum or np.max(widths) > maximum):
            reasons.add("lane_width_out_of_bounds")
            continue
        if geometry.reason:
            reasons.add(geometry.reason)
            continue
        mean_width = float(np.mean(widths))
        variation = float(np.ptp(widths) / mean_width)
        if variation > config.max_width_variation:
            reasons.add("lane_width_inconsistent")
            continue
        if config.max_curvature_m_inv is not None:
            if any(b.diagnostics is not None and b.diagnostics.max_curvature_px_inv / validated_mpp > config.max_curvature_m_inv
                   for b in (left, right)):
                reasons.add("metric_curvature_out_of_bounds")
                continue
        limited = left.limited_support or right.limited_support
        centre = replace(geometry.centreline, allow_extrapolation=not limited)
        rows = np.arange(np.ceil(centre.v_min), np.floor(centre.v_max) + 1)
        # Check only endpoint spans participating in this pair. Unsupported
        # candidate geometry outside the shared normal domain is not lane evidence
        # for this pair and must neither validate nor invalidate it.
        if (not _inside_roi(centre, rows, roi) or any(not _inside_roi(b.polynomial,
                np.arange(np.ceil(geometry.correspondences[:, column].min()),
                          np.floor(geometry.correspondences[:, column].max()) + 1), roi)
                for b, column in ((left, 1), (right, 3)))):
            reasons.add("pair_outside_supported_roi")
            continue
        score = min(left.confidence, right.confidence) * (1 - .35 * variation / config.max_width_variation)
        pair = LanePair(left.candidate_id, right.candidate_id, centre, float(score),
                              not (left.observed and right.observed), mean_width, variation,
                              mean_width * validated_mpp if validated_mpp is not None else None,
                              tuple(map(tuple, geometry.midpoints.tolist())),
                              tuple(map(tuple, geometry.correspondences.tolist())))
        pair = pair_observability(pair, left, right, config.row_step, config.normal_samples)
        far_intervals = tuple(sorted(set(left.unsupported_far_intervals + right.unsupported_far_intervals)))
        if limited and far_intervals:
            far_start = min(lo for lo, _ in far_intervals)
            far_intervals = ((far_start, centre.v_min),) if far_start < centre.v_min else ()
        pairs.append(replace(pair, limited_support=limited, valid_support_v_px=(centre.v_min, centre.v_max),
                             observed_span_px=pair.shared_observed_span_px, unsupported_far_intervals=far_intervals,
                             reason_codes=pair.reason_codes + (("limited_near_field",) if limited else ())))
    spatial_competition = len(pairs) > 1
    if len(pairs) > 1:
        disjoint = all(min(a.centreline.v_max,b.centreline.v_max) < max(a.centreline.v_min,b.centreline.v_min)
                       for a,b in itertools.combinations(pairs,2))
        if disjoint:
            spatial_competition = False
            reasons.add("disjoint_lane_support")
        else:
            reasons.add("multiple_compatible_pairs")
    ambiguous = crossing or spatial_competition
    return Association(tuple(pairs), ambiguous, tuple(sorted(reasons)))
