from dataclasses import asdict, replace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from camera_calibration.test_status import fixture_profile, report_for, measurements, ATTESTATION
from . import LaneConfig, LaneDetector
from .cli import main
from .synthetic import bev_fixture, scene
from .adversarial import arc_scene, circle_u


class CleanupTests(unittest.TestCase):
    def test_local_contrast_honors_white_floor(self):
        image = scene()
        image[image > 200] = 180
        strict = LaneDetector(LaneConfig(preprocessing="local_contrast", white_min_value=200)).process(bev_fixture(image))
        # The yellow boundary is still present; the dim neutral white must disappear.
        self.assertFalse(strict.valid)
        permissive = LaneDetector(LaneConfig(preprocessing="local_contrast", white_min_value=100)).process(bev_fixture(image))
        self.assertTrue(permissive.valid)

    def test_saturated_evidence_has_an_explicit_diagnostic(self):
        result = LaneDetector().process(bev_fixture(np.full((240, 320, 3), 255, np.uint8)))
        self.assertFalse(result.valid)
        self.assertIn("saturated_marking_evidence", result.reason_codes)
        self.assertGreater(result.evidence_fraction, .95)

    def test_orange_and_muted_tan_are_not_default_yellow_evidence(self):
        for color in ((0, 130, 255), (120, 170, 200)):
            image = np.full((240, 320, 3), 35, np.uint8)
            for u in (70, 170):
                cv2.line(image, (u, 6), (u, 233), color, 5)
            with self.subTest(color=color):
                self.assertFalse(LaneDetector().process(bev_fixture(image)).valid)

    def test_hough_can_be_disabled_without_geometry_loss(self):
        with patch("cv2.HoughLinesP", side_effect=AssertionError("disabled Hough")):
            result = LaneDetector(LaneConfig(use_hough=False)).process(bev_fixture(scene()))
        self.assertTrue(result.valid)

    def test_far_field_fork_stays_ambiguous(self):
        image = scene()
        cv2.line(image, (170, 105), (245, 10), (255, 255, 255), 5)
        result = LaneDetector().process(bev_fixture(image))
        self.assertFalse(result.valid)
        self.assertIsNone(result.selected_pair)

    def test_curved_prior_inference_uses_normal_offset(self):
        _, circle = arc_scene(700, 100, .6, (50, 190))
        rows = np.arange(50, 191, .2)
        image = np.full((240, 320, 3), 35, np.uint8)
        points = np.rint(np.column_stack((circle_u(rows, circle, 50), rows))).astype(np.int32)
        cv2.polylines(image, [points], False, (255, 255, 255), 5)
        cfg = LaneConfig(nominal_lane_width_px=100, single_boundary_role="left", min_lane_width_px=95, max_lane_width_px=105)
        result = LaneDetector(cfg).process(bev_fixture(image))
        self.assertFalse(result.valid)
        self.assertIsNotNone(result.selected_pair, result.reason_codes)
        self.assertTrue(result.selected_pair.inferred)
        self.assertAlmostEqual(result.selected_pair.width_mean_px, 100, delta=1)

    def test_invalid_cli_config_is_an_actionable_parse_error(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "bad.json"
            config.write_text('{"unrecognized_field": true}')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["--source", "0", "--profile", "unused.json", "--config", str(config)])
            self.assertEqual(error.exception.code, 2)

    def test_bev_metadata_and_signed_curvature_are_explicit(self):
        result = LaneDetector().process(bev_fixture(scene(((.001, -.24, 84.4), (.001, -.24, 184.4)))))
        self.assertEqual(result.bev_metadata.width_px, 320)
        self.assertEqual(result.bev_metadata.nominal_meters_per_pixel, .004)
        self.assertTrue(all(b.diagnostics.signed_curvature_reference_px_inv < 0 for b in result.boundaries))


class ValidationInputTests(unittest.TestCase):
    def test_attested_measurements_are_revalidated_not_loaded_as_trust(self):
        from .cli import _validation_from_measurements
        profile = fixture_profile()
        data = {"profile_fingerprint": profile.fingerprint(), "tolerance_m": .001,
                "provenance": asdict(ATTESTATION), "distances": [asdict(m) for m in measurements(profile)]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "measurements.json"
            path.write_text(json.dumps(data))
            report = _validation_from_measurements(path, profile)
            self.assertTrue(report.trusted_for(profile))
            data["profile_fingerprint"] = "forged"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                _validation_from_measurements(path, profile)
