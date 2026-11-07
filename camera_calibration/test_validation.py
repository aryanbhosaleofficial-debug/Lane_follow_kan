"""Tests for the calibration validator (SYNTHETIC data; every 'measurement' here is simulated)."""
import unittest

import numpy as np

from camera_calibration.bev import BEVConfig
from camera_calibration.errors import InvalidCalibrationInputError, ProfileError
from camera_calibration.intrinsics import IntrinsicParameters
from camera_calibration.profile import new_profile
from camera_calibration.synthetic import SyntheticPose, project_ground_points
from camera_calibration.test_bev import BEV, CAM, POSE, PROFILE, build_profile
from camera_calibration.test_profile import make_road_plane
from camera_calibration.validation import CalibrationValidator, DistanceMeasurement

VALIDATOR = CalibrationValidator(PROFILE)


def px(*ground):
    return [tuple(p) for p in project_ground_points(CAM, POSE, np.array(ground))]


class ConsistencyTests(unittest.TestCase):
    def test_correct_profile_passes_every_check(self):
        report = VALIDATOR.check_consistency()
        self.assertTrue(report.passed, str(report))
        names = {r.name for r in report.results}
        for expected in ("ground->image->ground", "ground->BEV->ground", "image->BEV->image", "BEV scale (1 m in pixels)",
                         "lens distort/undistort round trip", "BEV valid mask vs projection"):
            self.assertIn(expected, names)
        self.assertEqual(report.evidence, "self-consistency")
        self.assertIn("nothing about agreement with the real world", str(report))

    def test_validator_can_fail_on_a_broken_lens_model(self):
        # a distortion that folds back inside the image: undistortion cannot be inverted
        folded = IntrinsicParameters(640, 480, CAM.camera_matrix(), np.array([-1.0, 0, 0, 0, 0]), "standard5")
        base = new_profile(folded, "broken")
        profile = base.with_road_plane(make_road_plane(base)).with_bev(BEV)
        report = CalibrationValidator(profile).check_consistency()
        self.assertFalse(report.passed)
        self.assertTrue(any(not r.passed and "lens" in r.name for r in report.results), str(report))

    def test_requires_a_metric_profile(self):
        with self.assertRaises(ProfileError):
            CalibrationValidator(new_profile(CAM.intrinsics(), "bare"))

    def test_region_not_visible_is_reported_not_passed(self):
        # The lowest image ray (24.8 deg below the 28 deg optical axis = 52.8 deg) meets the road at 0.25/tan(52.8 deg) = 0.19 m,
        # so ground closer than that to the camera is out of view.
        behind = build_profile(bev=BEVConfig.from_extents(0.02, 0.12, -0.05, 0.05, 0.005))
        report = CalibrationValidator(behind).check_consistency()
        self.assertFalse(report.passed)
        self.assertEqual(report.results[0].name, "sample coverage")


class DistanceTests(unittest.TestCase):
    def test_exact_reference_lengths_pass(self):
        (a, b, c, d) = px([1.0, -0.25], [1.0, 0.25], [0.8, 0.0], [1.5, 0.0])
        report = VALIDATOR.check_distances([DistanceMeasurement(a, b, 0.50, "lane width"),
                                            DistanceMeasurement(c, d, 0.70, "along the lane")], tolerance_m=0.001)
        self.assertTrue(report.passed, str(report))
        self.assertEqual(report.evidence, "measured")
        self.assertEqual([r.name for r in report.results], ["lane width", "along the lane"])

    def test_wrong_reference_fails(self):
        a, b = px([1.0, -0.25], [1.0, 0.25])
        report = VALIDATOR.check_distances([DistanceMeasurement(a, b, 0.55)], tolerance_m=0.005)
        self.assertFalse(report.passed)
        self.assertAlmostEqual(report.results[0].value, 0.05, delta=1e-4)

    def test_noisy_calibration_stays_within_a_few_millimetres_inside_the_hull(self):
        noisy = CalibrationValidator(build_profile(noise=0.5, seed=4))
        a, b = px([1.0, -0.25], [1.0, 0.25])
        report = noisy.check_distances([DistanceMeasurement(a, b, 0.50)], tolerance_m=0.006)
        self.assertTrue(report.passed, str(report))

    def test_points_above_the_horizon_fail_cleanly(self):
        shallow = CalibrationValidator(build_profile(pose=SyntheticPose(0.25, 10.0)))
        report = shallow.check_distances([DistanceMeasurement((320.0, 5.0), (330.0, 50.0), 0.3)], tolerance_m=0.01)
        self.assertFalse(report.passed)
        self.assertIn("no ground position", report.results[0].detail)

    def test_extrapolated_endpoints_are_flagged(self):
        a, b = px([1.0, 0.0], [2.2, 0.0])                    # second point is beyond the calibrated hull (x up to 1.6 m + margin)
        report = VALIDATOR.check_distances([DistanceMeasurement(a, b, 1.2)], tolerance_m=0.05)
        self.assertIn("extrapolation", report.results[0].detail)

    def test_invalid_measurements_are_rejected(self):
        for kwargs in (dict(distance_m=0), dict(distance_m=-1), dict(distance_m=float("nan")), dict(distance_m="0.5"),
                       dict(distance_m=0.5, a_px=(float("nan"), 1.0)), dict(distance_m=0.5, a_px=(1.0,))):
            args = dict(a_px=(100.0, 300.0), b_px=(200.0, 300.0), **{k: v for k, v in kwargs.items() if k != "a_px"})
            if "a_px" in kwargs:
                args["a_px"] = kwargs["a_px"]
            with self.subTest(kwargs), self.assertRaises(InvalidCalibrationInputError):
                DistanceMeasurement(**args)
        with self.assertRaises(InvalidCalibrationInputError):
            VALIDATOR.check_distances([], 0.01)
        a, b = px([1.0, -0.25], [1.0, 0.25])
        for tol in (0, -1, float("nan")):
            with self.assertRaises(InvalidCalibrationInputError):
                VALIDATOR.check_distances([DistanceMeasurement(a, b, 0.5)], tol)


class HeldOutTests(unittest.TestCase):
    def test_independent_points_pass_and_wrong_ones_fail(self):
        truth = np.array([[0.9, -0.1], [1.2, 0.2], [1.4, -0.3]])
        pixels = project_ground_points(CAM, POSE, truth)
        self.assertTrue(VALIDATOR.check_held_out_points(pixels, truth, 0.001).passed)
        wrong = truth + [0.05, 0.0]
        report = VALIDATOR.check_held_out_points(pixels, wrong, 0.01)
        self.assertFalse(report.passed)
        self.assertAlmostEqual(report.results[0].value, 0.05, delta=1e-4)

    def test_outside_hull_is_noted(self):
        truth = np.array([[2.4, 0.0], [1.0, 0.0]])
        report = VALIDATOR.check_held_out_points(project_ground_points(CAM, POSE, truth), truth, 0.01)
        self.assertEqual(report.results[0].detail, "outside calibrated hull")
        self.assertEqual(report.results[1].detail, "")

    def test_input_validation(self):
        with self.assertRaises(InvalidCalibrationInputError):
            VALIDATOR.check_held_out_points([[1, 2]], [[1, 2], [3, 4]], 0.01)
        with self.assertRaises(InvalidCalibrationInputError):
            VALIDATOR.check_held_out_points([[1, 2]], [[1, 2]], 0)
        with self.assertRaises(InvalidCalibrationInputError):
            VALIDATOR.check_held_out_points([[np.nan, 2]], [[1, 2]], 0.1)


if __name__ == "__main__":
    unittest.main()
