"""Deterministic robust polynomial fitting on normalized BEV row coordinates."""
from __future__ import annotations

import numpy as np
from dataclasses import replace

from .config import LaneConfig
from .models import Boundary, FitDiagnostics, Polynomial


def _robust_model(t: np.ndarray, u: np.ndarray, degree: int, threshold: float):
    design = np.vander(t, degree + 1)
    full_coef = np.linalg.lstsq(design, u, rcond=None)[0]
    full_residual = np.abs(design @ full_coef - u)
    if np.all(full_residual <= threshold):
        return full_coef, np.ones(len(t), dtype=bool), full_residual, float(np.linalg.cond(design))
    rng = np.random.default_rng(37 + degree)
    subsets = np.stack([rng.choice(len(t), degree + 1, replace=False) for _ in range(48)])
    matrices = design[subsets]
    usable = np.linalg.cond(matrices) <= 1e5
    # Batched tiny solves avoid dozens of Python/SVD/lstsq calls per boundary.
    candidates = np.linalg.solve(matrices[usable], u[subsets[usable]][..., None])[..., 0]
    candidates = np.vstack((full_coef, candidates))
    residuals = np.abs(design @ candidates.T - u[:, None])
    masks = residuals <= threshold
    counts = masks.sum(axis=0)
    best_index = np.lexsort((np.median(residuals, axis=0), -counts))[0]
    inliers = masks[:, best_index]
    for _ in range(4):
        if inliers.sum() < degree + 2:
            return None
        matrix = design[inliers]
        condition = float(np.linalg.cond(matrix))
        if condition > 1e5:
            return None
        coef = np.linalg.lstsq(matrix, u[inliers], rcond=None)[0]
        residual = np.abs(design @ coef - u)
        updated = residual <= threshold
        if np.array_equal(inliers, updated):
            break
        inliers = updated
    # Refit exactly the final inlier set; report the same model's diagnostics.
    if inliers.sum() < degree + 2:
        return None
    coef = np.linalg.lstsq(design[inliers], u[inliers], rcond=None)[0]
    residual = np.abs(design @ coef - u)
    return coef, inliers, residual, float(np.linalg.cond(design[inliers]))


def support_intervals(rows: np.ndarray, step: int) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    rows = np.unique(rows)
    breaks = np.flatnonzero(np.diff(rows) > step * 1.5)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(rows) - 1]
    intervals = tuple((float(rows[a]), float(rows[b])) for a, b in zip(starts, ends))
    gaps = tuple((float(rows[k] + step / 2), float(rows[k + 1] - step / 2)) for k in breaks)
    return intervals, gaps


def _coherent_misfit(rows: np.ndarray, signed_errors: np.ndarray, config: LaneConfig):
    """Sustained signed bias is model error, not isolated raster/outlier noise."""
    order = np.argsort(rows)
    v, errors = rows[order], signed_errors[order]
    count = max(3, int(np.ceil(config.max_unmodeled_run_px / config.row_step)) + 1)
    if len(v) < count:
        return 0.0, ()
    means = np.convolve(errors, np.ones(count) / count, mode="valid")
    adjacent = np.convolve((np.diff(v) <= config.row_step * 1.5).astype(int), np.ones(count - 1, int), mode="valid") == count - 1
    good = adjacent & (np.abs(means) > config.near_field_max_error_px)
    merged: list[tuple[float, float]] = []
    for i in np.flatnonzero(good):
        lo, hi = float(v[i]), float(v[i + count - 1])
        if merged and lo <= merged[-1][1] + config.row_step:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    # One correlated window can be a stroke cap or raster phase. Require two
    # window spans of consistent bias before refusing a supported global model.
    intervals = tuple((lo, hi) for lo, hi in merged if hi - lo >= 2 * config.max_unmodeled_run_px)
    maximum = float(np.max(np.abs(means[adjacent]))) if np.any(adjacent) else 0.0
    return maximum, intervals


def fit_boundary(points: np.ndarray, config: LaneConfig, image_height: int, candidate_id: int, *, max_degree: int = 2) -> Boundary | None:
    """Return a supported fit or None. No endpoint extrapolation is performed.

    Random sampling has a fixed local seed and does not alter global RNG state.
    Row centres carry one vote each, so thick paint cannot dominate by pixel count.
    """
    p = np.asarray(points, dtype=float)
    if max_degree not in (1, 2):
        raise ValueError("max_degree must be 1 or 2")
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("fit points must be a finite Nx2 array of observed u,v")
    if len(p) < config.min_points:
        return None
    # Repeated rows are one observation, not independent support.
    rows = np.unique(p[:, 1])
    if len(rows) != len(p):
        p = np.array([(np.median(p[p[:, 1] == v, 0]), v) for v in rows])
    if len(p) < config.min_points:
        return None
    v, u = p[:, 1], p[:, 0]
    span = float(np.ptp(v))
    if span < config.min_span_px:
        return None
    origin = float((v.min() + v.max()) / 2)
    scale = span / 2
    t = (v - origin) / scale
    linear = _robust_model(t, u, 1, config.inlier_threshold_px)
    quadratic = (_robust_model(t, u, 2, config.inlier_threshold_px)
                 if max_degree == 2 and span >= config.quadratic_min_span_px else None)
    model, degree = linear, 1
    if quadratic is not None:
        if linear is None or (quadratic[1].sum() > linear[1].sum() + max(2, len(p) * .05)
                             or np.median(linear[2]) - np.median(quadratic[2]) > config.quadratic_improvement_px
                             or (np.max(linear[2][linear[1]]) > config.inlier_threshold_px * .75
                                 and np.median(quadratic[2]) < np.median(linear[2]))):
            model, degree = quadratic, 2
    if model is None:
        return None
    coef, inliers, residual, condition = model
    robust_count = int(inliers.sum())
    # Round raster/paint caps produce row centroids outside the centre-curve's
    # supported longitudinal domain. Trim endpoint support and refit, instead of
    # treating the cap tip as an extrapolated curve observation.
    accepted_v = v[inliers]
    inliers = inliers & (v >= accepted_v.min() + config.fit_endpoint_margin_px)
    inliers &= v <= accepted_v.max() - config.fit_endpoint_margin_px
    if inliers.sum() < config.min_points:
        return None
    design = np.vander(t, degree + 1)
    coef = np.linalg.lstsq(design[inliers], u[inliers], rcond=None)[0]
    residual = np.abs(design @ coef - u)
    condition = float(np.linalg.cond(design[inliers]))
    count = int(inliers.sum())
    outlier_fraction = 1 - robust_count / len(p)
    if count < config.min_points or outlier_fraction > config.max_outlier_fraction:
        return None
    vi = v[inliers]
    span = float(np.ptp(vi))
    coverage = min(1.0, len(np.unique(vi)) * config.row_step / image_height)
    observed_coverage = min(1.0, robust_count * config.row_step / image_height)
    rms = float(np.sqrt(np.mean(residual[inliers] ** 2)))
    intervals, gaps = support_intervals(vi, config.row_step)
    max_gap = max((b - a for a, b in gaps), default=0.0)
    if span < config.min_span_px or observed_coverage < config.min_coverage or rms > config.max_residual_px or max_gap > config.max_gap_px:
        return None
    normalized = np.poly1d(coef)
    pixel = normalized(np.poly1d([1 / scale, -origin / scale]))
    abc = np.pad(np.asarray(pixel.c, float), (3 - len(pixel.c), 0))
    poly = Polynomial(tuple(map(float, abc)), float(vi.min()), float(vi.max()))
    test_v = np.linspace(poly.v_min, poly.v_max, 64)
    slope = 2 * abc[0] * test_v + abc[1]
    # Include zero derivative, where quadratic curvature reaches its maximum.
    min_abs_slope = float(np.min(np.abs(slope)))
    if abc[0] != 0 and poly.v_min <= -abc[1] / (2 * abc[0]) <= poly.v_max:
        min_abs_slope = 0.0
    curvature = float(abs(2 * abc[0]) / (1 + min_abs_slope ** 2) ** 1.5)
    if np.max(np.abs(slope)) > config.max_abs_du_dv or curvature > config.max_curvature_px_inv:
        return None
    far = float(min(1.0, np.count_nonzero(vi < image_height / 2) * config.row_step / (image_height / 2)))
    near = float(min(1.0, np.count_nonzero(vi >= image_height / 2) * config.row_step / (image_height / 2)))
    quality = max(0.0, 1 - rms / config.max_residual_px)
    p95 = float(np.percentile(residual[inliers], 95))
    bias, misfit = _coherent_misfit(v[inliers], (design @ coef - u)[inliers], config)
    # The previous weighted score diluted fit quality to only 20% of confidence.
    # Enforce configured error tolerances on the whole score as well. Signed
    # bias must be sustained: a single dash cap is not systematic model error.
    # Errors outside a recovered near domain remain diagnostics, not local error.
    residual_quality = min(1.0, config.max_residual_px / max(p95, 1e-12),
                           config.near_field_max_error_px / max(bias, 1e-12) if misfit else 1.0)
    # Evidence density is conditional on the accepted fit domain. A side-exiting
    # marking can be well observed on a short domain; full-frame coverage remains
    # separately reported and still enforces the independent support floor.
    density = min(1.0, count * config.row_step / (span + config.row_step))
    confidence = float((.55 * density + .25 * min(1, span / image_height) + .20 * quality)
                       * (1 - outlier_fraction) * residual_quality)
    reasons = []
    if residual_quality < 1:
        reasons.append("fit_error_reduces_confidence")
    if gaps:
        reasons.append("unsupported_gaps")
    if coverage < .7:
        reasons.append("partial_coverage")
    if outlier_fraction > 0:
        reasons.append("outliers_removed")
    excluded = support_intervals(v[~inliers], config.row_step)[0] if np.any(~inliers) else ()
    if any(hi - lo >= config.max_unmodeled_run_px for lo, hi in excluded):
        reasons.append("model_support_truncated")
    if misfit:
        reasons.append("coherent_model_misfit")
    diagnostics = FitDiagnostics(degree, len(p), count, rms, p95,
                                 coverage, span, near, far, max_gap, curvature, condition,
                                 near_confidence=_section_score(residual[inliers][vi >= image_height / 2], near, config),
                                 far_confidence=_section_score(residual[inliers][vi < image_height / 2], far, config),
                                 input_span_px=float(np.ptp(v)), retained_fraction=float(count / len(p)),
                                 endpoint_trimmed_points=robust_count - count,
                                 observed_coverage_before_endpoint_trim=observed_coverage,
                                 all_observation_rms_px=float(np.sqrt(np.mean(residual ** 2))),
                                 all_observation_p95_px=float(np.percentile(residual, 95)),
                                 max_inlier_error_px=float(np.max(residual[inliers])),
                                 coherent_bias_max_px=bias,
                                 residual_quality_factor=float(residual_quality),
                                 signed_curvature_reference_px_inv=float(-2 * abc[0] /
                                     (1 + (2 * abc[0] * (poly.v_min + poly.v_max) / 2 + abc[1]) ** 2) ** 1.5))
    return Boundary(candidate_id, poly, True, confidence, tuple(reasons), intervals, gaps, diagnostics,
                    excluded_observation_intervals=excluded, model_misfit_intervals=misfit)


def fit_near_boundary(points: np.ndarray, config: LaneConfig, image_height: int, candidate_id: int, *,
                      minimum_v: float | None = None) -> Boundary | None:
    """Verify a near-field line anchored by independent observed span.

    Expand only while the anchored model agrees with observations; stop before
    the first far misfit, add a transition guard and refit within that domain.
    This is a limited observation, never an extrapolation through the turn.
    """
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("near-fit points must be finite Nx2 observed u,v")
    if len(p) < config.min_points:
        return None
    p = p[np.argsort(-p[:, 1], kind="stable")]
    required = max(config.min_span_px, config.min_pair_overlap_px) + 2 * config.fit_endpoint_margin_px
    ends = np.flatnonzero(p[0, 1] - p[:, 1] >= required)
    if not len(ends):
        return None
    stop = int(ends[0]) + 1
    anchor = p[:stop]
    origin = float(np.mean(anchor[:, 1]))
    scale = float(np.ptp(anchor[:, 1]) / 2)
    if scale <= 0:
        return None
    model = _robust_model((anchor[:, 1] - origin) / scale, anchor[:, 0], 1, config.near_field_max_error_px)
    if model is None or model[1].sum() < config.min_points or 1 - model[1].mean() > config.max_outlier_fraction:
        return None
    coef = np.poly1d(model[0])(np.poly1d([1 / scale, -origin / scale]))
    residual = np.abs(np.polyval(coef, p[:, 1]) - p[:, 0])
    bad = np.flatnonzero(residual[stop:] > config.near_field_max_error_px)
    if not len(bad) and minimum_v is None:
        return None  # no demonstrated unsupported far section
    lower = float(minimum_v) if minimum_v is not None else float(p[-1, 1])
    if len(bad):
        bad_index = stop + int(bad[0])
        lower = max(lower, float(p[bad_index, 1] + config.near_field_guard_px))
    prefix = p[p[:, 1] >= lower]
    local = replace(config, inlier_threshold_px=config.near_field_max_error_px,
                    max_residual_px=min(config.max_residual_px, config.near_field_max_error_px))
    fit = fit_boundary(prefix, local, image_height, candidate_id, max_degree=1)
    if fit is None or any(reason in fit.reason_codes for reason in ("model_support_truncated", "coherent_model_misfit")):
        return None
    poly = replace(fit.polynomial, allow_extrapolation=False)
    outside = (p[:, 1] < poly.v_min) | (p[:, 1] > poly.v_max)
    error = np.abs(np.polyval(poly.coefficients, p[:, 1]) - p[:, 0])
    far = ((float(p[:, 1].min()), poly.v_min),)
    diagnostics = replace(fit.diagnostics, observed_points=len(p), input_span_px=float(np.ptp(p[:, 1])),
                          retained_fraction=fit.diagnostics.inlier_points / len(p),
                          all_observation_rms_px=float(np.sqrt(np.mean(error ** 2))),
                          all_observation_p95_px=float(np.percentile(error, 95)))
    return replace(fit, polynomial=poly, limited_support=True, unsupported_far_intervals=far,
                   reason_codes=tuple(dict.fromkeys(fit.reason_codes + ("limited_near_field", "model_support_truncated"))),
                   diagnostics=diagnostics,
                   excluded_observation_intervals=support_intervals(p[outside, 1], config.row_step)[0])


def _section_score(residual: np.ndarray, coverage: float, config: LaneConfig) -> float:
    if len(residual) == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(residual ** 2)))
    return float(coverage * max(0.0, 1 - rms / config.max_residual_px))
