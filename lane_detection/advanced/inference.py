"""Optional same-frame width-prior inference; never temporal prediction."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from camera_calibration import BEVFrame
from camera_calibration.errors import ProfileError
from .config import LaneConfig
from .models import Boundary, Polynomial


def infer_missing_boundary(observed: Boundary, config: LaneConfig, frame: BEVFrame) -> Boundary | None:
    """Normal offset, followed by a bounded polynomial approximation.

    Refuse folded offsets or models that cannot represent the offset accurately.
    Metre priors require trusted M02 geometry; pixel priors remain pixel assumptions.
    Without an externally configured side, one line's role is unknowable.
    """
    if config.single_boundary_role is None:
        return None
    width = config.nominal_lane_width_px
    if config.nominal_lane_width_m is not None:
        try:
            frame.validated_geometry()
        except ProfileError:
            return None
        width = config.nominal_lane_width_m / frame.bev.meters_per_pixel
    if width is None:
        return None
    shift = width if config.single_boundary_role == "left" else -width
    source = observed.polynomial
    samples = source.samples(config.normal_samples)
    slope = 2 * source.coefficients[0] * samples[:, 1] + source.coefficients[1]
    samples[:, 0] += shift / np.sqrt(1 + slope ** 2)
    samples[:, 1] -= shift * slope / np.sqrt(1 + slope ** 2)
    if np.any(np.diff(samples[:, 1]) <= 0):
        return None
    v = samples[:, 1]
    scale, origin = float(np.ptp(v) / 2), float((v.min() + v.max()) / 2)
    if scale <= 0:
        return None
    coeff = np.linalg.lstsq(np.vander((v - origin) / scale, 3), samples[:, 0], rcond=None)[0]
    pixel = np.poly1d(coeff)(np.poly1d([1 / scale, -origin / scale]))
    poly = Polynomial(tuple(map(float, pixel.c)), float(v.min()), float(v.max()), source.allow_extrapolation)
    if np.max(np.abs(poly.evaluate(v) - samples[:, 0])) > config.centreline_fit_tolerance_px:
        return None
    return replace(observed, candidate_id=-1, polynomial=poly, observed=False,
                   confidence=observed.confidence * .45, reason_codes=("width_prior_inference",),
                   observed_intervals=(), unsupported_intervals=((poly.v_min, poly.v_max),),
                   diagnostics=None, ground_points_m=None, excluded_observation_intervals=())
