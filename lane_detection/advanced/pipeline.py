"""Synchronous M03 perception. Call from a consumer of M02's latest-only worker."""
from __future__ import annotations

import threading
import time
from dataclasses import replace
import numpy as np

from camera_calibration import BEVFrame, CalibrationStatus
from camera_calibration.errors import ProfileError, StaleFrameError
from .association import associate
from .config import LaneConfig
from .extraction import Extraction, extract
from .fitting import fit_boundary, fit_near_boundary
from .inference import infer_missing_boundary
from .models import BEVMetadata, Boundary, DebugImages, LaneObservation, StageTimings
from .preprocessing import Preprocessed, Preprocessor
from .validation import validate_frame


class LaneDetector:
    """Thread-safe, per-frame detector with one cached result; no temporal filter.

    Repeating the exact M02 BEVFrame returns the same observation. A different
    object claiming the latest source identity is rejected to prevent stale data.
    ``last_debug`` is only retained when requested; drawing is separate.
    """

    def __init__(self, config: LaneConfig | None = None):
        if config is not None and not isinstance(config, LaneConfig):
            raise TypeError("config must be LaneConfig")
        self.config = config or LaneConfig()
        self._preprocessor = Preprocessor(self.config)
        self._lock = threading.RLock()
        self._last_frame: BEVFrame | None = None
        self._last_observation: LaneObservation | None = None
        self._last_debug: DebugImages | None = None

    @property
    def last_debug(self) -> DebugImages | None:
        with self._lock:
            return self._last_debug

    def process(self, frame: BEVFrame, *, debug: bool = False) -> LaneObservation:
        with self._lock:
            if not isinstance(frame, BEVFrame):
                raise TypeError("M03 requires BEVFrame from M02")
            self._check_freshness(frame)  # also applies to cache hits
            if self._last_frame is not None and self._last_frame.key == frame.key:
                if self._last_frame is not frame:
                    raise ValueError("different BEVFrame objects claim the latest source identity")
                if debug and self._last_debug is None:
                    pre = self._preprocessor.process(frame)
                    self._last_debug = self._debug(pre, self._last_observation.boundaries,
                                                   extract(pre, self.config).hough_segments)
                return self._last_observation
            if (self._last_frame is not None and self._last_frame.source_session_id == frame.source_session_id
                    and frame.source_frame_id < self._last_frame.source_frame_id):
                raise ValueError("out-of-order source frame within a session")
            result = self._process(frame, debug)
            self._last_frame, self._last_observation = frame, result
            return result

    def _process(self, frame: BEVFrame, debug: bool) -> LaneObservation:
        start = time.perf_counter()
        validate_frame(frame)
        expected_mpp = self.config.configuration_meters_per_pixel
        if expected_mpp is not None and not np.isclose(expected_mpp, frame.bev.meters_per_pixel, rtol=1e-9, atol=0):
            raise ValueError("LaneConfig.for_bev grid scale does not match frame; rebuild configuration")
        metric = False
        reasons: list[str] = []
        if frame.calibration_status is CalibrationStatus.VALID_METRIC:
            try:
                frame.validated_geometry()
                metric = True
            except ProfileError:
                reasons.append("metric_trust_rejected")
        if not metric:
            reasons.append("pixel_geometry_only")
            if self.config.assumed_lane_width_m is not None:
                reasons.append("physical_configuration_unverified")
        pre = self._preprocessor.process(frame)
        if pre.saturated:
            reasons.append("saturated_marking_evidence")
        t_pre = time.perf_counter()
        extraction = Extraction((), (), False, False) if pre.saturated else extract(pre, self.config)
        t_extract = time.perf_counter()
        boundaries = []
        full_boundaries = []
        recovered = False
        for i, points in enumerate(() if extraction.overloaded else extraction.tracks):
            boundary = fit_boundary(points, self.config, frame.bev.height_px, i)
            if boundary is not None:
                boundary = self._width_evidence(boundary, points, extraction.track_widths[i])
                full_boundaries.append(boundary)
            if boundary is None or (boundary.marking_evidence_valid and any(reason in boundary.reason_codes for reason in ("model_support_truncated", "coherent_model_misfit"))):
                near = fit_near_boundary(points, self.config, frame.bev.height_px, i)
                if near is not None:
                    boundary, recovered = self._width_evidence(near, points, extraction.track_widths[i]), True
            if boundary is not None:
                if self.config.use_hough:
                    boundary = self._hough_support(boundary, extraction.hough_segments)
                boundaries.append(boundary)
        if recovered:
            # Verify both edges locally. A globally fitted partner can be biased
            # on the near straight even when its overall residual looks acceptable.
            minimum_v = max(b.polynomial.v_min for b in boundaries if b.limited_support)
            local_boundaries = []
            for boundary in boundaries:
                near = fit_near_boundary(extraction.tracks[boundary.candidate_id], self.config,
                                         frame.bev.height_px, boundary.candidate_id, minimum_v=minimum_v)
                if near is not None and boundary.marking_evidence_valid:
                    near = self._width_evidence(near, extraction.tracks[boundary.candidate_id], extraction.track_widths[boundary.candidate_id])
                    boundary = self._hough_support(near, extraction.hough_segments) if self.config.use_hough else near
                local_boundaries.append(boundary)
            boundaries = local_boundaries
        if len(boundaries) == 1 and boundaries[0].marking_evidence_valid:
            inferred = infer_missing_boundary(boundaries[0], self.config, frame)
            if inferred is not None:
                boundaries.append(inferred)
                reasons.append("inferred_boundary")
        if metric:
            boundaries = [replace(b, ground_points_m=tuple(map(tuple, frame.geometry.bev_to_ground(b.polynomial.samples()).tolist())) if b.marking_evidence_valid else None)
                          for b in boundaries]
        t_fit = time.perf_counter()
        association = associate(tuple(boundaries), pre.roi, self.config, frame.bev.meters_per_pixel if metric else None)
        reasons.extend(association.reason_codes)
        ambiguous = association.ambiguous or extraction.overloaded or extraction.branch_detected
        spatial_ambiguity = association.ambiguous or extraction.branch_detected
        if recovered:
            original = associate(tuple(full_boundaries), pre.roi, self.config, frame.bev.meters_per_pixel if metric else None)
            if original.ambiguous:
                ambiguous = True
                spatial_ambiguity = True
                reasons.extend(original.reason_codes)
        if any(any(reason in b.reason_codes for reason in ("model_support_truncated", "coherent_model_misfit"))
               and not b.limited_support for b in boundaries):
            ambiguous = True
            reasons.extend(sorted({reason for b in boundaries if not b.limited_support for reason in b.reason_codes
                                  if reason in ("model_support_truncated","coherent_model_misfit")}))
        if extraction.overloaded:
            reasons.append("candidate_capacity_exceeded")
            reasons.append("processing_budget_exceeded")
        if extraction.branch_detected:
            reasons.append("branch_geometry")
        pair = association.pairs[0] if len(association.pairs) == 1 and not ambiguous else None
        if pair is not None and not pair.inferred and pair.shared_observed_fraction == 0:
            reasons.append("no_shared_observed_support")
            if self.config.require_shared_observation:
                pair = None
        if pair is not None and not pair.inferred and pair.confidence >= self.config.min_pair_confidence:
            validity, score = "valid", pair.confidence
        elif boundaries or extraction.overloaded:
            validity = "uncertain"
            score = pair.confidence if pair is not None else min((b.confidence for b in boundaries), default=0.0) * .35
            if pair is not None and not pair.inferred:
                reasons.append("low_pair_confidence")
                pair = None  # no centreline output for an unsupported observed pair
            if spatial_ambiguity:
                reasons.append("ambiguous_geometry")
        else:
            validity, score = "invalid", 0.0
            reasons.append("no_supported_boundaries")
        observed_count = sum(b.observed for b in boundaries)
        if observed_count == 0:
            missing = ("left", "right")
        elif observed_count == 1:
            role = self.config.single_boundary_role
            missing = (("right" if role == "left" else "left"),) if role else ("unresolved_side",)
            reasons.append("single_observed_boundary")
        else:
            missing = ()
        if debug:
            self._last_debug = self._debug(pre, tuple(boundaries), extraction.hough_segments)
        else:
            self._last_debug = None
        end = time.perf_counter()
        timings = StageTimings((t_pre - start) * 1e3, (t_extract - t_pre) * 1e3,
                               (t_fit - t_extract) * 1e3, (end - t_fit) * 1e3, (end - start) * 1e3)
        result = LaneObservation(frame.source_session_id, frame.source_frame_id, frame.source_id,
                               frame.acquired_unix_ns, frame.acquired_monotonic_ns, frame.bev_sequence,
                               frame.profile_fingerprint, frame.calibration_status.value, metric, validity,
                               float(score), tuple(dict.fromkeys(reasons)), tuple(boundaries), association.pairs,
                               pair, missing, timings)
        reference = self.config.vehicle_reference_bev_px
        if self.config.ground_origin_is_vehicle:
            reference = (frame.bev.origin_u_px, frame.bev.origin_v_px)
        info = {"vehicle_reference_bev_px": reference, "processing_frame_age_s": self._check_freshness(frame),
                "max_frame_age_s": self.config.max_frame_age_s, "evidence_fraction": pre.evidence_fraction,
                "extraction_diagnostics": extraction.diagnostics,
                "bev_metadata": BEVMetadata(frame.bev.width_px, frame.bev.height_px, frame.bev.origin_u_px,
                    frame.bev.origin_v_px, frame.bev.meters_per_pixel, frame.geometry.road_plane.origin_description)}
        if pair is not None:
            info.update(limited_support=pair.limited_support,
                        support_status="limited_near_field" if pair.limited_support else "full_observed_domain",
                        usable_support_v_px=pair.valid_support_v_px, unsupported_far_intervals=pair.unsupported_far_intervals)
            if pair.limited_support:
                result = replace(result, reason_codes=result.reason_codes + ("limited_near_field", "far_field_unsupported"))
        if reference is None:
            info["reference_reason_codes"] = ("vehicle_reference_unspecified",)
        elif not (0 <= reference[0] <= frame.bev.width_px - 1
                  and 0 <= reference[1] <= frame.bev.height_px - 1):
            info["reference_reason_codes"] = ("vehicle_reference_outside_bev",)
        elif pair is None:
            info["reference_reason_codes"] = ("no_selected_pair",)
        else:
            u, v = reference
            by_id = {b.candidate_id: b for b in boundaries}
            curves = [by_id[pair.left_id].polynomial, by_id[pair.right_id].polynomial, pair.centreline]
            inside = all(p.v_min <= v <= p.v_max for p in curves)
            extension = max(0.0, max(v - p.v_max for p in curves))
            bounded = (not inside and all(v >= p.v_min for p in curves)
                       and extension <= self.config.max_reference_extension_px)
            values = [float(np.polyval(p.coefficients, v)) for p in curves]
            if bounded:
                points = np.array([(value, v) for value in values])
                bounded = self._preprocessor.reference_evidence(frame, points)
            if not inside and not bounded:
                info["reference_reason_codes"] = ("vehicle_reference_outside_observed_domain",)
            else:
                left_u, right_u, centre_u = values
                brackets = left_u <= u <= right_u
                offset = centre_u - u
                info.update(brackets_origin=brackets, ego_offset_px=offset,
                            ego_offset_m=offset * frame.bev.meters_per_pixel if metric else None,
                            lane_reference_valid=True, ego_lane_valid=result.valid and brackets)
                if bounded:
                    info.update(reference_extrapolated=True, reference_extension_px=extension,
                                reference_reason_codes=("bounded_near_reference_evidence",))
        final = time.perf_counter()
        return replace(result, **info, timings=replace(timings, total_ms=(final - start) * 1e3,
                       association_ms=timings.association_ms + (final - end) * 1e3))

    def _check_freshness(self, frame: BEVFrame) -> float:
        stamp = frame.acquired_monotonic_ns
        if isinstance(stamp, bool) or not isinstance(stamp, (int, np.integer)) or stamp <= 0:
            raise ValueError("acquired_monotonic_ns must be a positive integer")
        age = (time.monotonic_ns() - int(stamp)) / 1e9
        if age < -.01:
            raise ValueError("acquisition timestamp is in the future")
        limit = self.config.max_frame_age_s
        if limit is not None and age > limit:
            self._last_debug = None
            raise StaleFrameError(f"M03 source frame {frame.key} is {age:.3f}s old (limit {limit}s); use explicit offline replay policy if appropriate")
        return max(0.0, age)

    def _hough_support(self, boundary: Boundary, segments) -> Boundary:
        """Independent line votes support curve evidence but never gate curves."""
        poly = boundary.polynomial
        supported_rows = np.zeros(int(poly.v_max - poly.v_min) + 1, dtype=bool)
        for u1, v1, u2, v2 in segments:
            lo, hi = max(poly.v_min, v1), min(poly.v_max, v2)
            if hi <= lo:
                continue
            row = (lo + hi) / 2
            u = u1 + (row - v1) * (u2 - u1) / (v2 - v1)
            slope = (u2 - u1) / (v2 - v1)
            tangent = 2 * poly.coefficients[0] * row + poly.coefficients[1]
            if abs(u - float(poly.evaluate(row))) <= self.config.max_marking_width_px / 2 and abs(slope - tangent) <= .35:
                supported_rows[int(lo - poly.v_min):int(hi - poly.v_min) + 1] = True
        fraction = float(np.mean(supported_rows))
        diagnostics = replace(boundary.diagnostics, hough_support_fraction=fraction)
        reasons = boundary.reason_codes + (("hough_supported",) if fraction > .2 else ())
        return replace(boundary, diagnostics=diagnostics, reason_codes=reasons,
                       confidence=boundary.confidence * (.95 + .05 * fraction))

    def _width_evidence(self, boundary: Boundary, points: np.ndarray, widths: np.ndarray) -> Boundary:
        rows = points[:,1]
        core = np.zeros(len(rows),bool)
        # Remove cap rows of every observed dash, not just the frame endpoints.
        for lo,hi in boundary.observed_intervals:
            core |= (rows >= lo + self.config.row_step) & (rows <= hi - self.config.row_step)
        poly = boundary.polynomial
        residual = np.abs(np.polyval(poly.coefficients,rows)-points[:,0])
        core &= residual <= self.config.inlier_threshold_px
        if core.sum()<8:
            return boundary
        slope = 2*poly.coefficients[0]*rows[core]+poly.coefficients[1]
        normal_widths = widths[core]/np.sqrt(1+slope**2)
        lo,median,hi = map(float,np.percentile(normal_widths,[10,50,90]))
        ratio = hi/max(lo,1e-6)
        valid = ratio <= self.config.max_marking_width_ratio
        diagnostics = replace(boundary.diagnostics,marking_width_median_px=median,marking_width_p10_px=lo,
                              marking_width_p90_px=hi,marking_width_ratio=ratio)
        return replace(boundary,diagnostics=diagnostics,marking_evidence_valid=valid,
                       confidence=boundary.confidence if valid else 0.0,
                       reason_codes=boundary.reason_codes+(() if valid else ("marking_width_inconsistent",)))

    def _debug(self, pre: Preprocessed, boundaries: tuple[Boundary, ...], segments) -> DebugImages:
        accepted = np.zeros_like(pre.marking_mask)
        # Vectorized only over actual marking pixels; never draw a curve into evidence.
        v, u = np.nonzero(pre.marking_mask)
        good = np.zeros(len(v), dtype=bool)
        for b in boundaries:
            if b.observed and b.marking_evidence_valid:
                p = b.polynomial
                near = (v >= p.v_min) & (v <= p.v_max)
                near &= np.abs(u - p.evaluate(v)) <= self.config.max_marking_width_px / 2
                # Limit accepted observations to longitudinal intervals containing fit inliers.
                support = np.zeros(len(v), dtype=bool)
                for a, z in b.observed_intervals:
                    support |= (v >= a - self.config.row_step / 2) & (v <= z + self.config.row_step / 2)
                good |= near & support
        accepted[v[good], u[good]] = 255
        rejected = pre.marking_mask & ~accepted
        accepted.flags.writeable = False
        rejected.flags.writeable = False
        return DebugImages(pre.roi, pre.marking_mask, accepted, rejected, pre.edges, segments)
