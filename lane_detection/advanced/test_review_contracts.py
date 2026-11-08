from dataclasses import replace
import time
import unittest
import numpy as np

from camera_calibration.errors import StaleFrameError
from . import LaneConfig, LaneDetector
from .synthetic import bev_fixture, scene


class FreshnessTests(unittest.TestCase):
    def test_stale_replay_and_repeat_policies(self):
        frame = bev_fixture(scene())
        old = replace(frame, acquired_monotonic_ns=time.monotonic_ns() - 60_000_000_000)
        with self.assertRaises(StaleFrameError):
            LaneDetector().process(old)
        replay = LaneDetector(LaneConfig(max_frame_age_s=None)).process(old)
        self.assertTrue(replay.valid)
        detector = LaneDetector(LaneConfig(max_frame_age_s=1.0))
        first = detector.process(frame)
        self.assertTrue(first.valid)
        from unittest.mock import patch
        with patch("lane_detection.advanced.pipeline.time.monotonic_ns", return_value=frame.acquired_monotonic_ns + 2_000_000_000):
            with self.assertRaises(StaleFrameError):
                detector.process(frame)

    def test_invalid_timestamps(self):
        frame = bev_fixture(scene())
        for stamp in (0, -1, True, 1.5, time.monotonic_ns() + 10_000_000_000):
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                LaneDetector().process(replace(frame, acquired_monotonic_ns=stamp))


class EgoAndObservabilityTests(unittest.TestCase):
    def test_reference_is_explicit_and_independent_of_lane_validity(self):
        frame = bev_fixture(scene(((0, 0, 70), (0, 0, 200))))
        unknown = LaneDetector().process(frame)
        self.assertTrue(unknown.valid)
        self.assertIsNone(unknown.brackets_origin)
        self.assertFalse(unknown.lane_reference_valid)
        for reference, brackets, offset in (((160, 200), True, -25), ((250, 200), False, -115)):
            cfg = LaneConfig(vehicle_reference_bev_px=reference)
            result = LaneDetector(cfg).process(frame)
            self.assertTrue(result.valid)
            self.assertEqual(result.brackets_origin, brackets)
            self.assertEqual(result.ego_lane_valid, brackets)
            self.assertAlmostEqual(result.ego_offset_px, offset, delta=.5)
            self.assertIsNone(result.ego_offset_m)

    def test_bev_origin_must_be_explicitly_declared_vehicle_reference(self):
        frame = bev_fixture(scene())
        # Synthetic origin_v is 239, beyond observed boundary domain: do not extrapolate.
        result = LaneDetector(LaneConfig(ground_origin_is_vehicle=True)).process(frame)
        self.assertEqual(result.vehicle_reference_bev_px, (frame.bev.origin_u_px, frame.bev.origin_v_px))
        self.assertFalse(result.lane_reference_valid)
        self.assertIsNone(result.ego_offset_px)

    def test_pair_support_and_inference_are_explicit(self):
        full = LaneDetector().process(bev_fixture(scene()))
        dashed = LaneDetector().process(bev_fixture(scene(dashed=True)))
        a, b = full.selected_pair, dashed.selected_pair
        self.assertGreater(a.observed_fraction, .95)
        self.assertLess(b.observed_fraction, .75)
        self.assertTrue(b.unsupported_intervals)
        self.assertEqual(a.inferred_fraction, 0)
        self.assertGreater(a.shared_observed_span_px, b.shared_observed_span_px)
        inferred = LaneDetector(LaneConfig(nominal_lane_width_px=100, single_boundary_role="left")).process(
            bev_fixture(scene(((0, 0, 70),)))).selected_pair
        self.assertAlmostEqual(inferred.inferred_fraction, .5)
        self.assertEqual(inferred.shared_observed_fraction, 0)
        self.assertLessEqual(inferred.observed_fraction, .5)

    def test_quarter_observed_dashes_are_distinct_from_full_evidence(self):
        image = scene()
        keep = np.zeros(240, bool)
        for start in (6, 60, 114, 168, 222):
            keep[start:start + 12] = True
        image[~keep] = 35
        result = LaneDetector(LaneConfig(min_coverage=.20)).process(bev_fixture(image))
        self.assertTrue(result.valid, result.reason_codes)
        pair = result.selected_pair
        self.assertGreater(pair.observed_fraction, .18)
        self.assertLess(pair.observed_fraction, .38)
        self.assertEqual(pair.inferred_fraction, 0)
        self.assertIn("partial_pair_observation", pair.reason_codes)
        full = LaneDetector().process(bev_fixture(scene())).selected_pair
        self.assertGreater(full.confidence, pair.confidence)
