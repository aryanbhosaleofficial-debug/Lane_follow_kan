"""M03 screening parameters. Pixel limits refer to the configured BEV grid."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any

_MIN_RESOLUTION_SCALE = .25
_MAX_RESOLUTION_SCALE = 4.0


@dataclass(frozen=True)
class LaneConfig:
    """No measured physical lane width is assumed. Tune pixel limits for the grid.

    ``lane_width_m`` overrides pixel width bounds only with validated M02 geometry.
    Single-boundary inference requires both a prior and an explicit observed role.
    Confidence thresholds are engineering screening scores, not probabilities.
    """
    white_min_value: int = 140
    white_max_saturation: int = 85
    yellow_min_value: int = 90
    yellow_min_saturation: int = 110
    yellow_hue_min: int = 20
    yellow_hue_max: int = 40
    preprocessing: str = "color"          # color or local_contrast
    local_contrast: float = 12.0
    local_window: int = 31
    clahe: bool = False
    blur_size: int = 3                    # 1 disables Gaussian blur
    opening_size: int = 1                 # 1 disables opening
    roi_margin_px: int = 2
    canny_low: int = 40
    canny_high: int = 120
    hough_threshold: int = 12
    hough_min_length: int = 12
    hough_max_gap: int = 15
    use_hough: bool = field(default=False, kw_only=True)  # optional vote diagnostics; measured ablation
    max_hough_segments: int = field(default=64, kw_only=True)
    max_evidence_fraction: float = field(default=.5, kw_only=True)
    max_abs_du_dv: float = 1.5
    segment_duplicate_px: float = 5.0
    row_step: int = 3
    max_marking_width_px: int = 20
    association_margin_px: float = 9.0
    max_gap_px: int = 65
    max_tracks: int = 48
    max_row_runs: int = 32
    max_association_comparisons: int = field(default=4000, kw_only=True)  # spatially eligible pairs per frame
    max_fit_candidates: int = field(default=8, kw_only=True)
    min_points: int = 14
    min_span_px: int = 65
    min_coverage: float = 0.28
    max_residual_px: float = 3.0
    inlier_threshold_px: float = 4.0
    max_outlier_fraction: float = 0.30
    quadratic_min_span_px: int = 90
    quadratic_improvement_px: float = 0.35
    max_curvature_px_inv: float = 0.025
    min_pair_overlap_px: int = 65
    min_lane_width_px: float = 45.0
    max_lane_width_px: float = 260.0
    max_width_variation: float = 0.25
    min_pair_confidence: float = 0.45
    lane_width_m: tuple[float, float] | None = None
    max_curvature_m_inv: float | None = None
    nominal_lane_width_px: float | None = None
    nominal_lane_width_m: float | None = None
    single_boundary_role: str | None = None
    normal_samples: int = field(default=64, kw_only=True)
    normal_tangent_tolerance: float = field(default=.35, kw_only=True)  # radians
    centreline_fit_tolerance_px: float = field(default=1.0, kw_only=True)
    max_frame_age_s: float | None = field(default=.5, kw_only=True)  # None: replay
    vehicle_reference_bev_px: tuple[float, float] | None = field(default=None, kw_only=True)
    ground_origin_is_vehicle: bool = field(default=False, kw_only=True)
    max_unmodeled_run_px: int = field(default=12, kw_only=True)
    fit_endpoint_margin_px: float = field(default=3.0, kw_only=True)
    configuration_meters_per_pixel: float | None = field(default=None, kw_only=True)
    assumed_lane_width_m: tuple[float, float] | None = field(default=None, kw_only=True)
    near_field_max_error_px: float = field(default=1.0, kw_only=True)
    near_field_guard_px: int = field(default=12, kw_only=True)
    max_reference_extension_px: float = field(default=6.0, kw_only=True)
    max_marking_width_ratio: float = field(default=2.0, kw_only=True)
    require_shared_observation: bool = field(default=True, kw_only=True)

    @classmethod
    def for_bev(cls, bev: Any, lane_width_m: tuple[float, float], *,
                reference_meters_per_pixel: float = .004, require_validated_metric: bool = False,
                **overrides: Any) -> "LaneConfig":
        """Scale algorithm lengths from a reference grid, record physical assumptions.

        A BEVConfig alone cannot grant metric trust. The default width becomes
        NOMINAL pixel bounds with an explicit output limitation. Set
        require_validated_metric=True to refuse pairs until M02 grants trust.
        """
        from camera_calibration import BEVConfig
        if not isinstance(bev, BEVConfig):
            raise ValueError("bev must be BEVConfig")
        if isinstance(reference_meters_per_pixel, bool) or not isinstance(reference_meters_per_pixel, (int, float)) or not math.isfinite(reference_meters_per_pixel) or reference_meters_per_pixel <= 0:
            raise ValueError("reference_meters_per_pixel must be positive and finite")
        if not isinstance(require_validated_metric, bool):
            raise ValueError("require_validated_metric must be boolean")
        cls(lane_width_m=lane_width_m)
        base = cls()
        scale = reference_meters_per_pixel / bev.meters_per_pixel
        # Beyond 4x up/down sampling, rounding and paint footprints require a
        # separately tuned algorithm grid. Do not create huge kernels or imply
        # subpixel paint observations from an unreasonable automatic ratio.
        if not _MIN_RESOLUTION_SCALE <= scale <= _MAX_RESOLUTION_SCALE:
            raise ValueError(f"automatic BEV scale ratio {scale:g} must be within 0.25..4; configure a separately validated grid")
        lengths = ("roi_margin_px", "hough_min_length", "hough_max_gap", "row_step", "max_marking_width_px",
                   "max_gap_px", "min_span_px", "quadratic_min_span_px", "min_pair_overlap_px", "max_unmodeled_run_px", "near_field_guard_px")
        values = {name: max(1, int(round(getattr(base, name) * scale))) for name in lengths}
        for name in ("segment_duplicate_px", "association_margin_px", "max_residual_px", "inlier_threshold_px",
                     "quadratic_improvement_px", "centreline_fit_tolerance_px", "fit_endpoint_margin_px", "near_field_max_error_px", "max_reference_extension_px"):
            values[name] = getattr(base, name) * scale
        for name in ("blur_size", "opening_size", "local_window"):
            values[name] = 2 * int(round((getattr(base, name) - 1) * scale / 2)) + 1
        values.update(min_lane_width_px=lane_width_m[0] / bev.meters_per_pixel,
                      max_lane_width_px=lane_width_m[1] / bev.meters_per_pixel,
                      max_curvature_px_inv=base.max_curvature_px_inv / scale,
                      configuration_meters_per_pixel=bev.meters_per_pixel, assumed_lane_width_m=lane_width_m,
                      lane_width_m=lane_width_m if require_validated_metric else None)
        values.update(overrides)
        return replace(base, **values)

    def __post_init__(self) -> None:
        ints = {"white_min_value", "white_max_saturation", "yellow_min_value",
                "yellow_min_saturation", "yellow_hue_min", "yellow_hue_max", "local_window",
                "blur_size", "opening_size", "roi_margin_px", "canny_low", "canny_high",
                "hough_threshold", "hough_min_length", "hough_max_gap", "row_step", "max_hough_segments",
                "max_marking_width_px", "max_gap_px", "max_tracks", "max_row_runs", "max_association_comparisons", "max_fit_candidates",
                "min_points", "min_span_px", "quadratic_min_span_px", "min_pair_overlap_px", "normal_samples", "max_unmodeled_run_px", "near_field_guard_px"}
        optional = {"lane_width_m", "max_curvature_m_inv", "nominal_lane_width_px",
                    "nominal_lane_width_m", "single_boundary_role", "max_frame_age_s", "vehicle_reference_bev_px",
                    "configuration_meters_per_pixel", "assumed_lane_width_m"}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in optional or f.name in {"preprocessing", "clahe", "ground_origin_is_vehicle", "use_hough", "require_shared_observation"}:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"{f.name} must be finite and nonnegative")
            if f.name in ints and not isinstance(v, int):
                raise ValueError(f"{f.name} must be an integer")
            if v == 0 and f.name not in {"roi_margin_px", "canny_low", "hough_max_gap",
                                         "yellow_hue_min", "quadratic_improvement_px", "max_reference_extension_px"}:
                raise ValueError(f"{f.name} must be positive")
        for name in ("white_min_value", "white_max_saturation", "yellow_min_value",
                     "yellow_min_saturation", "canny_low", "canny_high"):
            if getattr(self, name) > 255:
                raise ValueError(f"{name} must be <= 255")
        if not 0 <= self.yellow_hue_min < self.yellow_hue_max <= 179:
            raise ValueError("yellow hue range must be ordered within 0..179")
        if self.canny_low >= self.canny_high:
            raise ValueError("canny_low must be less than canny_high")
        if self.min_lane_width_px >= self.max_lane_width_px:
            raise ValueError("pixel lane width bounds must be ordered")
        for name in ("blur_size", "opening_size", "local_window"):
            if getattr(self, name) % 2 != 1:
                raise ValueError(f"{name} must be odd")
        for name in ("min_coverage", "max_outlier_fraction", "max_width_variation", "min_pair_confidence", "max_evidence_fraction"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be in (0, 1)")
        if self.min_points < 3 or self.local_window < 3:
            raise ValueError("min_points and local_window must be at least 3")
        if (not isinstance(self.preprocessing, str) or self.preprocessing not in {"color", "local_contrast"}
                or not isinstance(self.clahe, bool)):
            raise ValueError("invalid preprocessing mode or clahe flag")
        if not isinstance(self.ground_origin_is_vehicle, bool):
            raise ValueError("ground_origin_is_vehicle must be a boolean")
        if not isinstance(self.use_hough, bool):
            raise ValueError("use_hough must be a boolean")
        if not isinstance(self.require_shared_observation, bool):
            raise ValueError("require_shared_observation must be a boolean")
        if self.max_marking_width_ratio < 1:
            raise ValueError("max_marking_width_ratio must be >= 1")
        if not 8 <= self.normal_samples <= 512:
            raise ValueError("normal_samples must be within 8..512")
        for name in optional - {"lane_width_m", "single_boundary_role", "vehicle_reference_bev_px", "assumed_lane_width_m"}:
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0):
                raise ValueError(f"{name} must be None or positive and finite")
        if self.vehicle_reference_bev_px is not None:
            p = self.vehicle_reference_bev_px
            if (not isinstance(p, tuple) or len(p) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                    or not math.isfinite(v) for v in p)):
                raise ValueError("vehicle_reference_bev_px must be a finite (u,v) tuple")
            if self.ground_origin_is_vehicle:
                raise ValueError("choose an explicit vehicle pixel reference OR the declared ground origin")
        for name in ("lane_width_m", "assumed_lane_width_m"):
            if getattr(self, name) is None:
                continue
            b = getattr(self, name)
            if (not isinstance(b, tuple) or len(b) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                    or not math.isfinite(v) for v in b) or not 0 < b[0] < b[1]):
                raise ValueError(f"{name} must be an ordered positive pair")
        if self.single_boundary_role is not None and (not isinstance(self.single_boundary_role, str)
                                                     or self.single_boundary_role not in {"left", "right"}):
            raise ValueError("single_boundary_role must be left, right or None")
        if self.nominal_lane_width_px is not None and self.nominal_lane_width_m is not None:
            raise ValueError("configure only one nominal width prior")
        if self.nominal_lane_width_px is not None and not self.min_lane_width_px <= self.nominal_lane_width_px <= self.max_lane_width_px:
            raise ValueError("nominal pixel width must lie inside width bounds")
        if self.nominal_lane_width_m is not None and self.lane_width_m is not None:
            if not self.lane_width_m[0] <= self.nominal_lane_width_m <= self.lane_width_m[1]:
                raise ValueError("nominal metric width must lie inside width bounds")
