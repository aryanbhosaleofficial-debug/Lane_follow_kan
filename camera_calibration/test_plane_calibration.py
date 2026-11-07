"""Tests for road-plane homography estimation (SYNTHETIC data; ground truth is known exactly)."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from camera_calibration.errors import (DegenerateGeometryError, InsufficientCalibrationDataError, InvalidCalibrationInputError)
from camera_calibration.plane_calibration import (RoadPlaneCalibrator, RoadPlaneConfig, decompose_plane_pose,
                                                  jacobian_determinants)
from camera_calibration.profile import CalibrationProfile, new_profile
from camera_calibration.roadplane import apply_homography, convex_hull, points_in_convex_polygon
from camera_calibration.synthetic import SyntheticCamera, SyntheticPose, project_ground_points, true_ground_homography
from camera_calibration.undistort import UndistortionSpec

CAM = SyntheticCamera(dist=(-0.28, 0.09, 0.001, -0.0008, -0.01))
INTR = CAM.intrinsics()
SPEC = UndistortionSpec.from_intrinsics(INTR, 0.0)
POSE = SyntheticPose(height_m=0.25, pitch_down_deg=28.0)
GX, GY = np.meshgrid(np.linspace(0.5, 1.6, 4), np.linspace(-0.35, 0.35, 3))
GROUND = np.column_stack([GX.ravel(), GY.ravel()])                      # 12 points
HX, HY = np.meshgrid(np.linspace(0.6, 1.5, 10), np.linspace(-0.3, 0.3, 7))
HELD_OUT = np.column_stack([HX.ravel(), HY.ravel()])                    # 70 points inside the hull, never used for fitting
FOUR = GROUND[[0, 3, 11, 8]]                                            # corners of the grid


def calibrator(**cfg):
    return RoadPlaneCalibrator(INTR, SPEC, RoadPlaneConfig(origin_description="SYNTHETIC origin", **cfg))


def pixels(pose=POSE, ground=GROUND, cam=CAM):
    return project_ground_points(cam, pose, ground)


def same_up_to_scale(A, B):
    return np.abs(A / np.linalg.norm(A) - B / np.linalg.norm(B)).max()


class HomographyBasicsTests(unittest.TestCase):
    def test_identity_homography_is_a_no_op(self):
        pts = np.array([[1.0, 2.0], [-3.5, 4.25]])
        out, valid = apply_homography(np.eye(3), pts)
        np.testing.assert_allclose(out, pts)
        self.assertTrue(valid.all())

    def test_known_perspective_transform(self):
        src = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        dst = np.float32([[10, 10], [200, 20], [180, 150], [5, 120]])
        H = cv2.getPerspectiveTransform(src, dst)
        H /= np.linalg.norm(H)
        out, valid = apply_homography(H, src)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(out, dst, atol=1e-3)
        # OpenCV's own perspectiveTransform agrees
        ref = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        np.testing.assert_allclose(out, ref, atol=1e-3)

    def test_invalid_denominators_are_nan_not_garbage(self):
        H = np.array([[1.0, 0, 0], [0, 1, 0], [1, 0, 0]])           # w = x
        pts = np.array([[1.0, 1.0], [0.0, 1.0], [-1.0, 1.0], [np.nan, 1.0]])
        out, valid = apply_homography(H, pts)
        self.assertEqual(valid.tolist(), [True, False, False, False])
        self.assertTrue(np.isnan(out[1:]).all())
        _, valid_any_sign = apply_homography(H, pts, require_positive_w=False)
        self.assertEqual(valid_any_sign.tolist(), [True, False, True, False])

    def test_convex_hull_and_membership(self):
        pts = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1], [1, 0.0]])
        hull = convex_hull(pts)
        self.assertEqual(len(hull), 4)
        inside = points_in_convex_polygon([[1, 1], [3, 1], [2.05, 1], [-0.2, -0.2]], hull, margin=0.1)
        self.assertEqual(inside.tolist(), [True, False, True, False])


class ExactRecoveryTests(unittest.TestCase):
    def test_noise_free_fit_recovers_homography_and_pose(self):
        cal = calibrator().estimate(pixels(), GROUND)
        self.assertLess(same_up_to_scale(cal.homography, true_ground_homography(POSE, SPEC.camera_matrix)), 1e-6)
        m = cal.metrics
        self.assertAlmostEqual(m["camera_height_m"], 0.25, delta=1e-6)
        self.assertAlmostEqual(m["camera_pitch_down_deg"], 28.0, delta=1e-5)
        self.assertAlmostEqual(m["camera_roll_deg"], 0.0, delta=1e-4)
        self.assertAlmostEqual(m["camera_yaw_deg"], 0.0, delta=1e-4)
        self.assertLess(m["max_reprojection_px"], 1e-3)
        self.assertLess(m["max_metric_error_m"], 1e-6)
        self.assertLess(m["orthonormality_error"], 1e-6)
        self.assertFalse(cal.exact_fit)
        self.assertEqual(cal.method, "least_squares")
        self.assertAlmostEqual(float(np.linalg.norm(cal.homography)), 1.0, places=12)

    def test_general_pose_with_roll_yaw_and_offset(self):
        pose = SyntheticPose(height_m=0.31, pitch_down_deg=22.0, yaw_deg=6.0, roll_deg=3.0, x_m=-0.2, y_m=0.05)
        cal = calibrator().estimate(pixels(pose), GROUND)
        m = cal.metrics
        for key, truth in (("camera_height_m", 0.31), ("camera_pitch_down_deg", 22.0), ("camera_roll_deg", 3.0),
                           ("camera_yaw_deg", 6.0), ("camera_x_m", -0.2), ("camera_y_m", 0.05)):
            with self.subTest(key):
                self.assertAlmostEqual(m[key], truth, delta=1e-4)

    def test_decompose_plane_pose_matches_true_pose_directly(self):
        info = decompose_plane_pose(SPEC.camera_matrix, true_ground_homography(POSE, SPEC.camera_matrix))
        self.assertAlmostEqual(info["camera_height_m"], 0.25, delta=1e-9)

    def test_four_points_are_exact_and_flagged(self):
        cal = calibrator().estimate(pixels(ground=FOUR), FOUR)
        self.assertTrue(cal.exact_fit)
        self.assertEqual(cal.method, "exact_4_point")
        self.assertTrue(any("exactly 4 points" in w for w in cal.warnings))
        self.assertLess(cal.metrics["max_reprojection_px"], 1e-3)
        self.assertNotIn("loo_rms_metric_error_m", cal.metrics)          # no redundancy -> no honest LOO figure
        self.assertLess(same_up_to_scale(cal.homography, true_ground_homography(POSE, SPEC.camera_matrix)), 1e-6)

    def test_four_point_fit_matches_opencv_perspective_transform(self):
        und = INTR.undistort_points(pixels(ground=FOUR), SPEC.camera_matrix)
        ref = cv2.getPerspectiveTransform(FOUR.astype(np.float32), und.astype(np.float32))
        cal = calibrator().estimate(pixels(ground=FOUR), FOUR)
        self.assertLess(same_up_to_scale(cal.homography, ref), 2e-4)      # float32 reference

    def test_leave_one_out_is_reported_with_redundancy(self):
        m = calibrator().estimate(pixels(), GROUND).metrics
        for key in ("loo_rms_metric_error_m", "loo_max_metric_error_m", "loo_rms_reprojection_px"):
            self.assertIn(key, m)

    def test_hull_matches_extreme_points(self):
        cal = calibrator().estimate(pixels(), GROUND)
        self.assertEqual(len(cal.calibrated_hull_m), 4)
        np.testing.assert_allclose(sorted(map(tuple, cal.calibrated_hull_m)), sorted(map(tuple, GROUND[[0, 3, 8, 11]])))


class NoisyFitTests(unittest.TestCase):
    def test_noisy_fit_generalises_inside_hull_but_not_outside(self):
        rng = np.random.default_rng(1)
        cal = calibrator()
        inside = []
        for _ in range(10):
            fit = cal.estimate(pixels() + rng.normal(0, 0.5, (len(GROUND), 2)), GROUND)
            inside.append(cal.validate(fit, pixels(ground=HELD_OUT), HELD_OUT)["rms_m"])
        self.assertLess(np.mean(inside), 0.004)           # measured: ~2 mm mean, ~5 mm worst over 30 draws
        self.assertLess(max(inside), 0.008)
        far = np.array([[2.0, 0.0], [2.5, 0.3], [3.0, -0.3]])
        report = cal.validate(fit, pixels(ground=far), far)
        self.assertEqual(report["inside_calibrated_hull"], [False, False, False])
        self.assertGreater(report["max_m"], 3 * np.mean(inside))         # extrapolation is visibly worse

    def test_points_space_must_be_labelled_correctly(self):
        cal = calibrator()
        right = cal.estimate(pixels(), GROUND, points_space="raw")
        wrong = cal.estimate(pixels(), GROUND, points_space="undistorted")    # raw pixels mislabelled
        err_right = cal.validate(right, pixels(ground=HELD_OUT), HELD_OUT)["rms_m"]
        err_wrong = cal.validate(wrong, pixels(ground=HELD_OUT), HELD_OUT)["rms_m"]
        self.assertLess(err_right, 1e-4)
        self.assertGreater(err_wrong, 0.01)                                    # measured ~40 mm; tens of times worse
        und = INTR.undistort_points(pixels(), SPEC.camera_matrix)
        again = cal.estimate(und, GROUND, points_space="undistorted")
        self.assertLess(same_up_to_scale(again.homography, right.homography), 1e-8)
        self.assertIsNone(again.image_points_raw_px)
        self.assertIsNotNone(right.image_points_raw_px)

    def test_ransac_rejects_gross_outliers(self):
        rng = np.random.default_rng(3)
        px = pixels() + rng.normal(0, 0.3, (len(GROUND), 2))
        px[[2, 7]] += [[45.0, -30.0], [-38.0, 40.0]]
        ls = calibrator().estimate(px, GROUND)
        for method in ("ransac", "lmeds"):
            with self.subTest(method):
                fit = calibrator(method=method).estimate(px, GROUND)
                self.assertFalse(fit.inlier_mask[2] or fit.inlier_mask[7])
                self.assertEqual(int(fit.inlier_mask.sum()), 10)
                self.assertTrue(any("rejected as outliers" in w for w in fit.warnings))
                robust = calibrator().validate(fit, pixels(ground=HELD_OUT), HELD_OUT)["rms_m"]
                naive = calibrator().validate(ls, pixels(ground=HELD_OUT), HELD_OUT)["rms_m"]
                self.assertLess(robust, 0.005)
                self.assertGreater(naive, 3 * robust)
                self.assertEqual(len(fit.ground_points_m), 12)               # all evidence retained, only the mask differs

    def test_robust_method_with_four_points_falls_back_with_a_warning(self):
        fit = calibrator(method="ransac").estimate(pixels(ground=FOUR), FOUR)
        self.assertTrue(fit.exact_fit)
        self.assertTrue(any("needs more than 4" in w for w in fit.warnings))

    def test_inconsistent_intrinsics_are_flagged_by_the_pose_check(self):
        wrong_cam = SyntheticCamera(fx=700.0, fy=520.0, dist=CAM.dist)            # focal length in x is 35 % off
        wrong = RoadPlaneCalibrator(wrong_cam.intrinsics(), UndistortionSpec.from_intrinsics(wrong_cam.intrinsics(), 0.0),
                                    RoadPlaneConfig())
        fit = wrong.estimate(pixels(), GROUND)
        self.assertGreater(fit.metrics["orthonormality_error"], 0.10)
        self.assertTrue(any("not consistent with the intrinsics" in w for w in fit.warnings))


class RejectionTests(unittest.TestCase):
    def test_too_few_points(self):
        with self.assertRaises(InsufficientCalibrationDataError):
            calibrator().estimate(pixels(ground=GROUND[:3]), GROUND[:3])

    def test_input_validation(self):
        good = pixels()
        cases = {
            "count mismatch": (good[:-1], GROUND),
            "nan pixel": (np.where(np.eye(12, 2) > 0, np.nan, good), GROUND),
            "inf ground": (good, np.where(np.eye(12, 2) > 0, np.inf, GROUND)),
            "wrong dims": (np.zeros((12, 3)), GROUND),
            "strings": ([["a", "b"]] * 12, GROUND),
            "far outside image": (good + 5000, GROUND),
            "not 2d": (np.zeros(5), np.zeros(5)),
        }
        for label, (img, gnd) in cases.items():
            with self.subTest(label), self.assertRaises(InvalidCalibrationInputError):
                calibrator().estimate(img, gnd)

    def test_degenerate_layouts(self):
        line_y = np.column_stack([np.linspace(0.5, 1.5, 6), np.zeros(6)])
        cases = {
            "all collinear": line_y,
            "three of four collinear": np.array([[0.6, 0.0], [1.0, 0.0], [1.4, 0.0], [1.0, 0.3]]),
            "repeated ground point": np.vstack([GROUND[:5], GROUND[:1]]),
            "nearly identical ground points": np.vstack([GROUND[:5], GROUND[:1] + 1e-6]),
        }
        for label, gnd in cases.items():
            with self.subTest(label), self.assertRaises(DegenerateGeometryError):
                calibrator().estimate(pixels(ground=gnd), gnd)

    def test_repeated_image_point(self):
        px = pixels()
        px[5] = px[4]
        with self.assertRaisesRegex(DegenerateGeometryError, "identical"):
            calibrator().estimate(px, GROUND)

    def test_mirrored_and_swapped_axes_are_rejected(self):
        px = pixels()
        flipped_y = GROUND * [1, -1]
        flipped_x = GROUND * [-1, 1]
        swapped = GROUND[:, ::-1]
        for label, g in (("y flipped", flipped_y), ("x flipped", flipped_x), ("x/y swapped", swapped)):
            with self.subTest(label), self.assertRaises(DegenerateGeometryError):
                calibrator().estimate(px, g)
        with self.assertRaisesRegex(DegenerateGeometryError, "mirrored"):
            calibrator().estimate(px, flipped_y)

    def test_mirror_check_cannot_be_bypassed_by_the_facing_setting(self):
        # handedness is independent of the camera heading, so 'unchecked' must not disable it
        with self.assertRaisesRegex(DegenerateGeometryError, "mirrored"):
            calibrator(camera_facing="unchecked").estimate(pixels(), GROUND * [1, -1])

    def test_mirror_check_is_heading_independent_for_a_rear_camera(self):
        rear = SyntheticPose(0.25, 28.0, yaw_deg=180.0)
        back = GROUND * [-1, 1]                                    # the road behind the vehicle
        fit = calibrator(camera_facing="rear").estimate(pixels(rear, back), back)
        self.assertAlmostEqual(abs(fit.metrics["camera_yaw_deg"]), 180.0, delta=1e-3)      # +-180 are the same heading
        self.assertTrue((jacobian_determinants(fit.homography, back) < 0).all())      # same sign as a forward camera
        with self.assertRaisesRegex(DegenerateGeometryError, "mirrored"):
            calibrator(camera_facing="rear").estimate(pixels(rear, back), back * [1, -1])

    def test_declared_heading_must_match_recovered_heading(self):
        rear = SyntheticPose(0.25, 28.0, yaw_deg=180.0)
        back = GROUND * [-1, 1]
        with self.assertRaisesRegex(DegenerateGeometryError, "optical axis"):
            calibrator(camera_facing="forward").estimate(pixels(rear, back), back)
        with self.assertRaisesRegex(DegenerateGeometryError, "optical axis"):
            calibrator(camera_facing="rear").estimate(pixels(), GROUND)

    def test_rotated_axis_labelling_is_caught_by_the_heading_check(self):
        # X measured sideways: a proper rotation (not a mirror) of the ground frame by 90 degrees
        rotated = np.column_stack([GROUND[:, 1], -GROUND[:, 0]])
        px = pixels()
        with self.assertRaisesRegex(DegenerateGeometryError, "optical axis|rotated"):
            calibrator().estimate(px, rotated)
        fit = calibrator(camera_facing="unchecked").estimate(px, rotated)         # allowed when stated explicitly
        self.assertAlmostEqual(abs(fit.metrics["camera_yaw_deg"]), 90.0, delta=1e-3)

    def test_small_toe_in_is_accepted(self):
        yawed = SyntheticPose(0.25, 28.0, yaw_deg=20.0)
        fit = calibrator().estimate(pixels(yawed), GROUND)
        self.assertAlmostEqual(fit.metrics["camera_yaw_deg"], 20.0, delta=1e-3)

    def test_points_straddling_the_horizon_plane_are_rejected(self):
        ground = np.vstack([GROUND, [[-0.5, 0.1], [-0.4, -0.2]]])     # behind the camera: negative depth
        H = true_ground_homography(POSE, SPEC.camera_matrix)
        und, _ = apply_homography(H, ground, require_positive_w=False)
        with self.assertRaisesRegex(DegenerateGeometryError, "denominators change sign"):
            calibrator().estimate(und, ground, points_space="undistorted")

    def test_condition_number_limit_is_enforced(self):
        with self.assertRaisesRegex(DegenerateGeometryError, "ill-conditioned"):
            calibrator(max_condition_number=1.5).estimate(pixels(), GROUND)

    def test_config_validation(self):
        for kw in ({"method": "magic"}, {"camera_facing": "left"}, {"max_condition_number": 0}, {"ransac_threshold_px": -1},
                   {"min_hull_area_fraction": 1.0}, {"orthonormality_warning": float("nan")}, {"origin_description": 5}):
            with self.subTest(kw), self.assertRaises(InvalidCalibrationInputError):
                RoadPlaneConfig(**kw)

    def test_undistortion_size_must_match(self):
        with self.assertRaises(InvalidCalibrationInputError):
            RoadPlaneCalibrator(INTR, UndistortionSpec(SPEC.camera_matrix, 0.0, (320, 240)))

    def test_jacobian_sign_is_negative_for_forward_camera(self):
        H = true_ground_homography(POSE, SPEC.camera_matrix)
        self.assertTrue((jacobian_determinants(H, HELD_OUT) < 0).all())


class ProfileIntegrationTests(unittest.TestCase):
    def test_estimated_road_plane_survives_profile_save_and_load(self):
        rng = np.random.default_rng(5)
        fit = calibrator().estimate(pixels() + rng.normal(0, 0.4, (12, 2)), GROUND)
        profile = new_profile(INTR, "synthetic-cam").with_road_plane(fit)
        with tempfile.TemporaryDirectory() as tmp:
            back = CalibrationProfile.load(profile.save(Path(tmp) / "p.json"))
        np.testing.assert_allclose(back.road_plane.homography, fit.homography, rtol=1e-14)
        np.testing.assert_array_equal(back.road_plane.inlier_mask, fit.inlier_mask)
        self.assertEqual(back.road_plane.metrics.keys(), fit.metrics.keys())
        self.assertIn("SYNTHETIC origin", back.road_plane.summary())

    def test_road_plane_for_a_different_undistortion_is_refused_by_the_profile(self):
        fit = calibrator().estimate(pixels(), GROUND)
        with self.assertRaises(Exception):
            new_profile(INTR, "c", alpha=1.0).with_road_plane(fit)


if __name__ == "__main__":
    unittest.main()
