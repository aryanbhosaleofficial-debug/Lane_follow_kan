import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from camera_calibration import BEVProcessor, CalibrationStatus, ProcessorConfig
from camera_calibration.mapper import CameraGeometry
from camera_calibration.test_bev import PROFILE, STRAIGHT_IMG
from camera_calibration.test_pipeline import packet, RoadCapture
from camera_calibration.test_status import fixture_profile, report_for
from camera_input import CameraConfig, CameraManager
from . import LaneConfig, LaneDetector
from .cli import main
from .synthetic import bev_fixture, scene
from .visualization import draw_observation


class PipelineTests(unittest.TestCase):
    def test_straight_identity_centre_and_no_unverified_metres(self):
        f = bev_fixture(scene())
        result = LaneDetector().process(f)
        self.assertTrue(result.valid)
        self.assertEqual((result.source_session_id, result.source_frame_id), f.key)
        self.assertEqual(result.acquired_unix_ns, f.acquired_unix_ns)
        self.assertEqual(result.acquired_monotonic_ns, f.acquired_monotonic_ns)
        self.assertLess(abs(float(result.selected_pair.centreline.evaluate(120)) - 120), .5)
        self.assertFalse(result.metric_validated)
        self.assertEqual(result.calibration_status, "METRIC_UNVERIFIED")
        self.assertIsNone(result.selected_pair.width_mean_m)
        self.assertTrue(all(b.ground_points_m is None for b in result.boundaries))
        json.dumps(result.to_dict(), allow_nan=False)

    def test_repeat_is_same_observation_and_conflicting_repeat_rejected(self):
        detector = LaneDetector()
        f = bev_fixture(scene())
        result = detector.process(f)
        self.assertIs(detector.process(f, debug=True), result)
        self.assertIsNotNone(detector.last_debug)
        with self.assertRaises(ValueError):
            detector.process(replace(f))
        detector.process(bev_fixture(scene(), 2))
        with self.assertRaises(ValueError):
            detector.process(f)

    def test_no_temporal_prediction_when_evidence_disappears(self):
        detector = LaneDetector()
        solid = detector.process(bev_fixture(scene(), 1))
        dashed = detector.process(bev_fixture(scene(dashed=True), 2))
        single = detector.process(bev_fixture(scene(((0, 0, 70),)), 3))
        empty = detector.process(bev_fixture(scene(()), 4))
        self.assertTrue(dashed.valid)
        self.assertGreater(solid.confidence, dashed.confidence)
        self.assertGreater(dashed.confidence, single.confidence)
        self.assertGreater(single.confidence, empty.confidence)
        self.assertIsNone(single.selected_pair)
        self.assertEqual(single.missing_boundaries, ("unresolved_side",))
        self.assertEqual(empty.validity, "invalid")
        self.assertIsNone(empty.selected_pair)

    def test_inferred_pair_remains_uncertain(self):
        result = LaneDetector(LaneConfig(nominal_lane_width_px=100, single_boundary_role="left")).process(
            bev_fixture(scene(((0, 0, 70),))))
        self.assertFalse(result.valid)
        self.assertEqual(result.validity, "uncertain")
        self.assertTrue(result.selected_pair.inferred)
        self.assertEqual(result.missing_boundaries, ("right",))
        self.assertEqual(sum(b.observed for b in result.boundaries), 1)

    def test_multiple_lanes_and_junction_do_not_select_driving_lane(self):
        for abc in (((0, 0, 50), (0, 0, 150), (0, 0, 250)),
                    ((0, .55, 50), (0, -.55, 180))):
            with self.subTest(abc=abc):
                result = LaneDetector().process(bev_fixture(scene(abc)))
                self.assertFalse(result.valid)
                self.assertIsNone(result.selected_pair)

    def test_debug_evidence_subset_and_overlay_do_not_modify_input(self):
        detector = LaneDetector()
        f = bev_fixture(scene(dashed=True))
        before = f.image.copy()
        result = detector.process(f, debug=True)
        debug = detector.last_debug
        self.assertFalse(np.any((debug.accepted_mask != 0) & (debug.candidate_mask == 0)))
        np.testing.assert_array_equal(debug.accepted_mask | debug.rejected_mask, debug.candidate_mask)
        self.assertEqual(np.count_nonzero(debug.accepted_mask[45:55]), 0)
        overlay = draw_observation(f.image, result, debug)
        self.assertFalse(np.shares_memory(overlay, f.image))
        np.testing.assert_array_equal(before, f.image)

    def test_untrusted_metric_constraints_and_forged_status(self):
        cfg = LaneConfig(lane_width_m=(.38, .42))
        f = bev_fixture(scene())
        for frame in (f, replace(f, calibration_status=CalibrationStatus.VALID_METRIC)):
            r = LaneDetector(cfg).process(frame)
            self.assertFalse(r.valid)
            self.assertFalse(r.metric_validated)
            self.assertIsNone(r.selected_pair)
            self.assertIn("metric_constraints_unavailable", r.reason_codes)

    def test_simulated_trust_gate_exports_correct_ground_axes(self):
        # SIMULATED neutral-origin fixture solely exercises software trust; not
        # physical evidence. Same technique as existing M02 status tests.
        f = bev_fixture(scene())
        profile = fixture_profile().with_bev(f.bev)
        trusted = BEVProcessor(profile, ProcessorConfig(max_frame_age_s=None), report_for(profile)).process(packet())
        f = replace(f, geometry=trusted.geometry, profile_fingerprint=trusted.profile_fingerprint,
                    calibration_status=trusted.calibration_status,
                    _validation_report=trusted._validation_report, _trust_policy=trusted._trust_policy)
        r = LaneDetector(LaneConfig(lane_width_m=(.38, .42))).process(f)
        self.assertTrue(r.valid)
        self.assertTrue(r.metric_validated)
        self.assertAlmostEqual(r.selected_pair.width_mean_m, .4, places=3)
        left = next(b for b in r.boundaries if b.candidate_id == r.selected_pair.left_id)
        ground = np.asarray(left.ground_points_m)
        np.testing.assert_allclose(ground[:, 1], .36, atol=.003)
        self.assertGreater(ground[0, 0], ground[-1, 0])  # v down means X decreases
        unverified = LaneDetector().process(bev_fixture(scene()))
        # Pixel evidence score is independent of whether physical scale is trusted.
        self.assertAlmostEqual(r.confidence, unverified.confidence, places=9)

    def test_capacity_limit_cannot_return_a_confident_lane(self):
        result = LaneDetector(LaneConfig(max_tracks=2)).process(
            bev_fixture(scene(((0, 0, 50), (0, 0, 150), (0, 0, 250)))))
        self.assertFalse(result.valid)
        self.assertIsNone(result.selected_pair)
        self.assertIn("candidate_capacity_exceeded", result.reason_codes)

    def test_hough_votes_support_but_do_not_gate_curve_fits(self):
        frame = bev_fixture(scene())
        supported = LaneDetector(LaneConfig(use_hough=True)).process(frame)
        with patch("cv2.HoughLinesP", return_value=None):
            no_hough = LaneDetector(LaneConfig(use_hough=True)).process(frame)
        self.assertTrue(no_hough.valid)
        self.assertLess(no_hough.confidence, supported.confidence)
        self.assertTrue(all(b.diagnostics.hough_support_fraction == 0 for b in no_hough.boundaries))

    def test_actual_m01_m02_m03_contract(self):
        with CameraManager(CameraConfig(source=0), RoadCapture) as camera:
            packet_m01 = camera.wait_for_frame(timeout=1)
            before = packet_m01.image.copy()
            frame = BEVProcessor(PROFILE).process(packet_m01)
            result = LaneDetector().process(frame)
            self.assertEqual(result.source_frame_id, packet_m01.frame_id)
            self.assertEqual(result.source_session_id, packet_m01.session_id)
            self.assertEqual(result.acquired_monotonic_ns, packet_m01.acquired_monotonic_ns)
            self.assertGreaterEqual(len(result.boundaries), 1)
            np.testing.assert_array_equal(packet_m01.image, before)
            self.assertIs(camera.latest(), camera.latest())

    def test_headless_cli_image_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "road.png"
            cv2.imwrite(str(image), STRAIGHT_IMG)
            profile = root / "profile.json"
            PROFILE.save(profile)
            out = root / "out"
            with patch("cv2.imshow", side_effect=AssertionError("headless display")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--source", str(image), "--profile", str(profile),
                                       "--headless", "--output", str(out)]), 0)
            records = (out / "observations.jsonl").read_text().splitlines()
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0])["source_frame_id"], 1)
            self.assertEqual(len(list(out.glob("*.png"))), 4)
            self.assertIsNotNone(cv2.imread(str(out / "000000_overlay.png")))
            summary = json.loads((out / "run_summary.json").read_text())
            self.assertTrue(summary["completed_to_eof"])
            self.assertEqual(summary["processed_observations"], 1)


if __name__ == "__main__":
    unittest.main()
