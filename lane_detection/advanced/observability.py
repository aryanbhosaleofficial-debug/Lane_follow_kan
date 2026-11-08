"""Pair evidence measured on uniform bins of the accepted centreline v domain."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .models import Boundary, LanePair


def _observed(rows: np.ndarray, boundary: Boundary, row_step: int) -> np.ndarray:
    mask = np.zeros(len(rows), bool)
    if boundary.observed:
        for lo, hi in boundary.observed_intervals:
            mask |= (rows >= lo - row_step / 2) & (rows <= hi + row_step / 2)
    return mask


def pair_observability(pair: LanePair, left: Boundary, right: Boundary, row_step: int, count: int) -> LanePair:
    """Denominator: 2*count endpoints, uniformly sampled at centre-domain bin centres.

    Endpoint v values are interpolated from actual normal correspondences, not
    assumed equal. Shared span sums bin widths where BOTH endpoints are observed.
    Missing paint and inferred geometry are kept distinct.
    """
    points = np.asarray(pair.normal_correspondences_px)
    mid_v = (points[:, 1] + points[:, 3]) / 2
    lo, hi = pair.centreline.v_min, pair.centreline.v_max
    edges = np.linspace(lo, hi, count + 1)
    rows = (edges[:-1] + edges[1:]) / 2
    lv = np.interp(rows, mid_v, points[:, 1])
    rv = np.interp(rows, mid_v, points[:, 3])
    a, b = _observed(lv, left, row_step), _observed(rv, right, row_step)
    both = a & b
    gaps = ~both
    changes = np.diff(np.r_[False, gaps, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    intervals = tuple((float(edges[start]), float(edges[end])) for start, end in zip(starts, ends))
    observed = float((a.sum() + b.sum()) / (2 * count))
    inferred = ((not left.observed) + (not right.observed)) / 2
    reasons = []
    if observed < .95:
        reasons.append("partial_pair_observation")
    if inferred:
        reasons.append("inferred_pair")
    if not inferred and not np.any(both):
        reasons.append("no_shared_observed_support")
    return replace(pair, observed_fraction=observed, inferred_fraction=float(inferred),
                   shared_observed_fraction=float(both.mean()), shared_observed_span_px=float(both.mean() * (hi - lo)),
                   unsupported_intervals=intervals, support_samples=count, reason_codes=tuple(reasons))
