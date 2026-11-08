"""Normal intersections of fitted boundaries; no equal-row width approximation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import LaneConfig
from .models import Polynomial


@dataclass(frozen=True, eq=False)
class NormalGeometry:
    correspondences: np.ndarray
    midpoints: np.ndarray
    widths: np.ndarray
    centreline: Polynomial | None
    reason: str | None


def _intersections(source: Polynomial, target: Polynomial, rows: np.ndarray,
                   direction: int, tolerance: float):
    """Intersect target with source normal: (q-p) dot (u'(v),1) == 0.

    Quadratic roots are solved with a cancellation-resistant formula. Multiple
    directed roots in the target's supported domain are ambiguous, not guessed.
    """
    a, b, c = target.coefficients
    s = 2 * source.coefficients[0] * rows + source.coefficients[1]
    u = source.evaluate(rows)
    aa, bb, cc = s * a, 1 + s * b, s * (c - u) - rows
    roots = np.full((len(rows), 2), np.nan)
    linear = np.abs(aa) < 1e-12
    usable = linear & (np.abs(bb) > 1e-12)
    roots[usable, 0] = -cc[usable] / bb[usable]
    discriminant = bb ** 2 - 4 * aa * cc
    quadratic = ~linear & (discriminant >= 0)
    indices = np.flatnonzero(quadratic)
    q = -.5 * (bb[indices] + np.copysign(np.sqrt(discriminant[indices]), bb[indices]))
    roots[indices, 0] = q / aa[indices]
    nz = np.abs(q) > 1e-12
    roots[indices[nz], 1] = cc[indices[nz]] / q[nz]
    target_u = target.evaluate(roots)
    good = np.isfinite(roots) & (roots >= target.v_min) & (roots <= target.v_max)
    good &= direction * (target_u - u[:, None]) > 0
    count = good.sum(axis=1)
    if np.any(count > 1):
        return None, "ambiguous_normal_intersection"
    indices = np.flatnonzero(count == 1)
    root = roots[indices, np.argmax(good[indices], axis=1)]
    other_u = target.evaluate(root)
    other_s = 2 * a * root + b
    compatible = np.abs(np.arctan(other_s) - np.arctan(s[indices])) <= tolerance
    p = np.column_stack((u[indices], rows[indices]))
    qpoint = np.column_stack((other_u, root))
    return np.column_stack((p[compatible], qpoint[compatible])), None


def normal_geometry(left: Polynomial, right: Polynomial, config: LaneConfig) -> NormalGeometry:
    correspondences = []
    for source, target, direction in ((left, right, 1), (right, left, -1)):
        rows = np.linspace(source.v_min, source.v_max, config.normal_samples)
        points, reason = _intersections(source, target, rows, direction, config.normal_tangent_tolerance)
        if reason:
            return NormalGeometry(np.empty((0, 4)), np.empty((0, 2)), np.empty(0), None, reason)
        if len(points):
            correspondences.append(points if direction == 1 else points[:, [2, 3, 0, 1]])
    if not correspondences:
        return NormalGeometry(np.empty((0, 4)), np.empty((0, 2)), np.empty(0), None, "no_supported_normal_correspondence")
    points = np.vstack(correspondences)
    midpoints = (points[:, :2] + points[:, 2:]) / 2
    order = np.argsort(midpoints[:, 1], kind="stable")
    points, midpoints = points[order], midpoints[order]
    widths = np.linalg.norm(points[:, :2] - points[:, 2:], axis=1)
    v = midpoints[:, 1]
    span = float(np.ptp(v))
    if span < config.min_pair_overlap_px or len(v) < 8:
        return NormalGeometry(points, midpoints, widths, None, "insufficient_normal_support")
    origin, scale = (v.min() + v.max()) / 2, span / 2
    design = np.vander((v - origin) / scale, 3)
    coeff = np.linalg.lstsq(design, midpoints[:, 0], rcond=None)[0]
    curve = np.poly1d(coeff)(np.poly1d([1 / scale, -origin / scale]))
    poly = Polynomial(tuple(map(float, curve.c)), float(v.min()), float(v.max()))
    error = np.max(np.abs(poly.evaluate(v) - midpoints[:, 0]))
    if error > config.centreline_fit_tolerance_px:
        return NormalGeometry(points, midpoints, widths, None, "centreline_model_unsupported")
    return NormalGeometry(points, midpoints, widths, poly, None)
