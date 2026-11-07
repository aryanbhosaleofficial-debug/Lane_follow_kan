"""Trust-state tests use simulated measurements, never physical-camera evidence."""
import unittest
import tempfile
import json
from pathlib import Path
from argparse import Namespace
from dataclasses import replace

from camera_calibration import (BEVProcessor, CalibrationStatus, CalibrationValidator,
                                DistanceMeasurement, ProfileError)
from camera_calibration.test_bev import PROFILE
from camera_calibration.test_pipeline import packet


class CalibrationStatusTests(unittest.TestCase):
    def test_unverified_profile_cannot_offer_validated_geometry(self):
        frame = BEVProcessor(PROFILE).process(packet())
        self.assertEqual(frame.calibration_status, CalibrationStatus.METRIC_UNVERIFIED)
        self.assertEqual(frame.profile_fingerprint, PROFILE.fingerprint())
        with self.assertRaises(ProfileError):
            frame.validated_geometry()

    def test_measured_report_must_match_profile(self):
        validator = CalibrationValidator(PROFILE)
        geometry = validator.geometry
        a, b = geometry.ground_to_image([[1.0, -0.1], [1.0, 0.1]])
        report = validator.check_distances([DistanceMeasurement(tuple(a), tuple(b), 0.2)], 0.001)
        self.assertTrue(report.passed)
        frame = BEVProcessor(PROFILE, validation_report=report).process(packet())
        self.assertEqual(frame.calibration_status, CalibrationStatus.VALID_METRIC)
        self.assertIs(frame.validated_geometry(), frame.geometry)
        with self.assertRaises(ProfileError):
            BEVProcessor(PROFILE, validation_report=replace(report, profile_fingerprint="wrong"))
        with self.assertRaises(ProfileError):
            BEVProcessor(PROFILE, validation_report=validator.check_consistency())

    def test_cli_report_is_bound_to_the_profile(self):
        from camera_calibration.cli import cmd_check_distances
        geometry = CalibrationValidator(PROFILE).geometry
        a, b = geometry.ground_to_image([[1.0, -0.1], [1.0, 0.1]])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = PROFILE.save(root / "profile.json")
            measurements = root / "measurements.json"
            measurements.write_text(json.dumps({"distances": [{"a_px": a.tolist(), "b_px": b.tolist(),
                                                                "distance_m": 0.2}]}), encoding="utf-8")
            output = root / "report.json"
            code = cmd_check_distances(Namespace(profile=str(profile), measurements=str(measurements),
                                                  tolerance_m=0.001, report_out=str(output)))
            self.assertEqual(code, 0)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(saved["passed"])
            self.assertEqual(saved["profile_fingerprint"], PROFILE.fingerprint())


if __name__ == "__main__": unittest.main()
