"""Trust-state tests use simulated measurements, never physical-camera evidence."""
import unittest
import tempfile
import json
from pathlib import Path
from argparse import Namespace
from dataclasses import replace

import numpy as np

from camera_calibration import (BEVProcessor, CalibrationStatus, CalibrationValidator,
                                DistanceMeasurement, MeasurementProvenance, ProfileError)
from camera_calibration.errors import InvalidCalibrationInputError
from camera_calibration.validation import (MAX_TRUSTED_TOLERANCE_M, MIN_TRUSTED_REFERENCES, CheckResult,
                                           ValidationReport, TrustPolicy)
from camera_calibration.pipeline import ProcessorConfig
from camera_calibration.test_bev import PROFILE
from camera_calibration.test_pipeline import packet

ATTESTATION = MeasurementProvenance("test operator", "ruler", "test fixture", True)
# Three simulated "tape measurements": lateral 0.2 m, lateral 0.4 m and longitudinal 0.5 m.
SEGMENTS = (((1.0, -0.1), (1.0, 0.1)), ((0.9, -0.2), (0.9, 0.2)), ((0.8, 0.0), (1.3, 0.0)))


def fixture_profile(origin="test fixture"):
    """SIMULATED profile with a deliberately neutral origin text, only to exercise the trust gate."""
    return PROFILE.with_road_plane(replace(PROFILE.road_plane, origin_description=origin))


def measurements(profile, count=3):
    geometry = CalibrationValidator(profile).geometry
    out = []
    for p, q in SEGMENTS[:count]:
        a, b = geometry.ground_to_image([p, q])
        out.append(DistanceMeasurement(tuple(a), tuple(b), float(np.hypot(q[0] - p[0], q[1] - p[1]))))
    return out


def report_for(profile, count=3, tolerance=0.001, provenance=ATTESTATION):
    return CalibrationValidator(profile).check_distances(measurements(profile, count), tolerance, provenance=provenance)


class CalibrationStatusTests(unittest.TestCase):
    def test_unverified_profile_cannot_offer_validated_geometry(self):
        frame = BEVProcessor(PROFILE).process(packet())
        self.assertEqual(frame.calibration_status, CalibrationStatus.METRIC_UNVERIFIED)
        self.assertEqual(frame.profile_fingerprint, PROFILE.fingerprint())
        with self.assertRaises(ProfileError):
            frame.validated_geometry()

    def test_measured_report_must_match_profile(self):
        # This is simulated data with a deliberately renamed fixture for testing the
        # trust gate. It is not evidence of physical-camera calibration.
        fixture = fixture_profile()
        validator = CalibrationValidator(fixture)
        report = report_for(fixture)
        self.assertTrue(report.passed)
        frame = BEVProcessor(fixture, validation_report=report).process(packet())
        self.assertEqual(frame.calibration_status, CalibrationStatus.VALID_METRIC)
        self.assertIs(frame.validated_geometry(), frame.geometry)
        with self.assertRaises(ProfileError):
            BEVProcessor(fixture, validation_report=replace(report, profile_fingerprint="wrong"))
        with self.assertRaises(ProfileError):
            BEVProcessor(fixture, validation_report=validator.check_consistency())
        with self.assertRaises(ProfileError):
            BEVProcessor(PROFILE, validation_report=report_for(PROFILE))          # origin text says SYNTHETIC
        with self.assertRaises(ProfileError):
            BEVProcessor(PROFILE, validation_report=report)                       # report belongs to another profile
        with self.assertRaises(ProfileError):
            BEVProcessor(fixture, validation_report=replace(report, results=()))
        with self.assertRaises(ProfileError):
            BEVProcessor(fixture, validation_report=report_for(fixture, provenance=None))

    def test_failed_report_cannot_grant_trust(self):
        fixture = fixture_profile()
        wrong = [replace(m, distance_m=m.distance_m * 1.5) for m in measurements(fixture)]
        report = CalibrationValidator(fixture).check_distances(wrong, 0.001, provenance=ATTESTATION)
        self.assertFalse(report.passed)
        with self.assertRaises(ProfileError):
            BEVProcessor(fixture, validation_report=report)

    def test_cli_report_is_bound_to_the_profile(self):
        from camera_calibration.cli import cmd_check_distances
        geometry = CalibrationValidator(PROFILE).geometry
        a, b = geometry.ground_to_image([[1.0, -0.1], [1.0, 0.1]])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = PROFILE.save(root / "profile.json")
            measurements_file = root / "measurements.json"
            measurements_file.write_text(json.dumps({"distances": [{"a_px": a.tolist(), "b_px": b.tolist(),
                                                                     "distance_m": 0.2}]}), encoding="utf-8")
            output = root / "report.json"
            code = cmd_check_distances(Namespace(profile=str(profile), measurements=str(measurements_file),
                                                  tolerance_m=0.001, report_out=str(output)))
            self.assertEqual(code, 0)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(saved["passed"])
            self.assertEqual(saved["profile_fingerprint"], PROFILE.fingerprint())
            # The saved JSON must say, in a field, that it cannot grant trust and which profile origin it saw.
            self.assertIs(saved["grants_trusted_geometry"], False)
            self.assertEqual(saved["profile_origin_description"], PROFILE.road_plane.origin_description)
            self.assertIn("review record", saved["trust_note"])

    def test_constructed_summary_cannot_grant_trusted_geometry(self):
        fixture = fixture_profile()
        provenance = MeasurementProvenance("test", "ruler", "fixture", True)
        forged = ValidationReport(tuple(CheckResult(f"distance {i}", True, 0.0, 0.01, "m") for i in range(3)),
                                  "measured", profile_fingerprint=fixture.fingerprint(), provenance=provenance)
        with self.assertRaises(ProfileError):
            BEVProcessor(fixture, validation_report=forged)

    def test_uniform_rescale_cannot_inherit_validation(self):
        fixture = fixture_profile()
        proc = BEVProcessor(fixture, ProcessorConfig(allow_uniform_rescale=True), report_for(fixture))
        small = packet(image=np.zeros((240, 320, 3), np.uint8))
        frame = proc.process(small)
        self.assertEqual(frame.calibration_status, CalibrationStatus.METRIC_UNVERIFIED)
        self.assertEqual(frame.profile_fingerprint, frame.geometry.profile.fingerprint())
        with self.assertRaises(ProfileError):
            frame.validated_geometry()


class EvidenceFloorTests(unittest.TestCase):
    """M02-SONNET-REV-002: a passing report is not automatically trustworthy evidence."""

    def test_floor_constants_are_documented_values(self):
        self.assertEqual(MIN_TRUSTED_REFERENCES, 3)
        self.assertEqual(MAX_TRUSTED_TOLERANCE_M, 0.05)

    def test_too_few_references_pass_but_are_not_trusted(self):
        fixture = fixture_profile()
        for count in (1, 2):
            report = report_for(fixture, count=count)
            self.assertTrue(report.passed)
            self.assertFalse(report.trusted_for(fixture), f"{count} reference(s)")
            with self.assertRaises(ProfileError):
                BEVProcessor(fixture, validation_report=report)
        self.assertTrue(report_for(fixture, count=3).trusted_for(fixture))

    def test_a_vacuous_tolerance_cannot_certify_wrong_distances(self):
        fixture = fixture_profile()
        absurd = [replace(m, distance_m=37.0) for m in measurements(fixture)]       # 37 m claimed for ~0.2-0.5 m segments
        report = CalibrationValidator(fixture).check_distances(absurd, 1e6, provenance=ATTESTATION)
        self.assertTrue(report.passed)                                               # the operator chose the tolerance...
        self.assertFalse(report.trusted_for(fixture))                                # ...but it cannot buy trust
        with self.assertRaises(ProfileError):
            BEVProcessor(fixture, validation_report=report)

    def test_tolerance_boundary(self):
        fixture = fixture_profile()
        # Absolute limits also need to be meaningful relative to the measured lengths.
        report = CalibrationValidator(fixture).check_distances(measurements(fixture), MAX_TRUSTED_TOLERANCE_M,
                    provenance=ATTESTATION, trust_policy=TrustPolicy(max_relative_error=0.3))
        self.assertTrue(report.trusted_for(fixture))
        self.assertFalse(report_for(fixture, tolerance=MAX_TRUSTED_TOLERANCE_M * 1.01).trusted_for(fixture))

    def test_repeated_references_cannot_buy_coverage(self):
        fixture = fixture_profile()
        m = measurements(fixture)[0]
        with self.assertRaisesRegex(ValueError, "duplicate distance"):
            CalibrationValidator(fixture).check_distances([m, m, m], 0.001, provenance=ATTESTATION)
        truth = np.array([[1.13, 0.07]] * 3)
        validator = CalibrationValidator(fixture)
        with self.assertRaisesRegex(ValueError, "duplicate held-out"):
            validator.check_held_out_points(validator.geometry.ground_to_image(truth), truth, 0.001)

    def test_huge_custom_absolute_limit_still_needs_relative_precision(self):
        fixture = fixture_profile()
        report = CalibrationValidator(fixture).check_distances(measurements(fixture), 1e6,
                    provenance=ATTESTATION, trust_policy=TrustPolicy(max_tolerance_m=1e6))
        self.assertFalse(report.trusted_for(fixture))

    def test_local_cluster_cannot_certify_whole_region(self):
        fixture = fixture_profile()
        validator = CalibrationValidator(fixture)
        truth = np.array([[1.001, 0.001], [1.002, 0.003], [1.003, 0.001]])
        report = validator.check_held_out_points(validator.geometry.ground_to_image(truth), truth, 1e-5,
                                                provenance=ATTESTATION)
        self.assertTrue(report.passed)
        self.assertFalse(report.trusted_for(fixture))

    def test_held_out_points_need_the_same_floor(self):
        fixture = fixture_profile()
        validator = CalibrationValidator(fixture)
        truth = np.array([[1.13, 0.07], [0.93, -0.21], [1.3, 0.2]])
        pixels = validator.geometry.ground_to_image(truth)
        many = validator.check_held_out_points(pixels, truth, 0.005, provenance=ATTESTATION)
        self.assertTrue(many.trusted_for(fixture))
        one = validator.check_held_out_points(pixels[:1], truth[:1], 0.005, provenance=ATTESTATION)
        self.assertTrue(one.passed)
        self.assertFalse(one.trusted_for(fixture))                                   # 1 point + its RMS row is not 3 references

    def test_synthetic_marker_cannot_be_hidden_by_spacing_or_synonym(self):
        for origin in ("SYNTHETIC", "S Y N T H E T I C", "synthetic-rig", "simulated rig", "placeholder", "fake tape", "Mock"):
            profile = fixture_profile(origin)
            self.assertFalse(report_for(profile).trusted_for(profile), origin)
        self.assertTrue(report_for(fixture_profile("floor tape, corner A")).trusted_for(fixture_profile("floor tape, corner A")))

    def test_measurement_on_the_hull_boundary_is_not_rejected_by_float_noise(self):
        fixture = fixture_profile()
        vertex = tuple(float(v) for v in fixture.road_plane.calibrated_hull_m[0])       # exact calibration-hull corner
        interior = (1.0, 0.0)
        geometry = CalibrationValidator(fixture).geometry
        a, b = geometry.ground_to_image([vertex, interior])
        distance = float(np.hypot(vertex[0] - interior[0], vertex[1] - interior[1]))
        report = CalibrationValidator(fixture).check_distances([DistanceMeasurement(tuple(a), tuple(b), distance)], 0.001)
        self.assertTrue(report.passed, str(report))


class ConsumerPolicyTests(unittest.TestCase):
    """M01M02-SONNET-001: the consumer, not the report's producer, decides how strict acceptance is."""
    LOOSE = TrustPolicy(max_tolerance_m=1e9, max_relative_error=0.99, min_span_fraction=1e-9)

    def loose_report(self, fixture):
        # computed lengths are ~0.2 / 0.4 / 0.5 m; the "measured" lengths are 0 %, ~48 % and ~38 % off
        wrong = [replace(m, distance_m=d) for m, d in zip(measurements(fixture), (0.2, 0.21, 0.31))]
        return CalibrationValidator(fixture).check_distances(wrong, 0.19, provenance=ATTESTATION, trust_policy=self.LOOSE)

    def test_a_producer_chosen_loose_policy_cannot_grant_trust(self):
        fixture = fixture_profile()
        report = self.loose_report(fixture)
        self.assertTrue(report.passed)
        with self.assertRaises(ProfileError):
            BEVProcessor(fixture, validation_report=report)                          # default consumer policy applies
        self.assertFalse(report.trusted_for(fixture, TrustPolicy()))

    def test_a_consumer_can_explicitly_choose_a_looser_or_stricter_policy(self):
        fixture = fixture_profile()
        frame = BEVProcessor(fixture, validation_report=self.loose_report(fixture), trust_policy=self.LOOSE).process(packet())
        self.assertEqual(frame.calibration_status, CalibrationStatus.VALID_METRIC)   # a deliberate, visible consumer decision
        self.assertIs(frame.validated_geometry(), frame.geometry)
        with self.assertRaises(ProfileError):                                         # 4 references demanded, 3 supplied
            BEVProcessor(fixture, validation_report=report_for(fixture), trust_policy=TrustPolicy(min_references=4))

    def test_validated_geometry_rechecks_the_consumer_policy(self):
        fixture = fixture_profile()
        frame = BEVProcessor(fixture, validation_report=report_for(fixture)).process(packet())
        self.assertIs(frame.validated_geometry(), frame.geometry)
        stricter = replace(frame, _trust_policy=TrustPolicy(max_tolerance_m=0.0005))
        with self.assertRaises(ProfileError):
            stricter.validated_geometry()
        with self.assertRaises(ProfileError):
            replace(frame, _trust_policy=None).validated_geometry()

    def test_bad_arguments_are_controlled_errors(self):
        fixture = fixture_profile()
        for bad in (None, "profile.json", {}, object()):
            with self.assertRaises(ProfileError, msg=repr(bad)):
                BEVProcessor(bad)
        with self.assertRaises(InvalidCalibrationInputError):
            BEVProcessor(fixture, trust_policy="strict")


if __name__ == "__main__": unittest.main()
