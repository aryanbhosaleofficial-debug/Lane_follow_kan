"""Tests for M02 intrinsic calibration. All data is SYNTHETIC (see synthetic.py)."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from camera_calibration.errors import (CalibrationQualityError, InsufficientCalibrationDataError,
                                       InvalidCalibrationInputError)
from camera_calibration.intrinsics import (DISTORTION_MODELS, IntrinsicCalibrationConfig, IntrinsicCalibrator,
                                           IntrinsicParameters, chessboard_object_points)
from camera_calibration.synthetic import (SyntheticCamera, project_board_corners, random_board_poses,
                                          render_chessboard)

PATTERN, SQUARE_M = (9, 6), 0.025
STD_CAMERA = SyntheticCamera(dist=(-0.28, 0.09, 0.001, -0.0008, -0.01))
FISH_CAMERA = SyntheticCamera(fx=330.0, fy=330.0, dist=(0.04, -0.01, 0.002, -0.0005), model="fisheye")


def config(**kw):
    base = dict(pattern_size=PATTERN, square_size=25, square_unit="mm")
    base.update(kw)
    return IntrinsicCalibrationConfig(**base)


def detections_for(camera, count=14, noise=0.1, seed=0, **pose_kw):
    rng = np.random.default_rng(seed)
    out = []
    for i, (R, t) in enumerate(random_board_poses(camera, PATTERN, SQUARE_M, count, seed=seed, **pose_kw)):
        c = project_board_corners(camera, PATTERN, SQUARE_M, R, t)
        out.append((f"view{i:02d}", c + rng.normal(0, noise, c.shape)))
    return out


class ConfigValidationTests(unittest.TestCase):
    def test_valid_config_converts_units_to_metres(self):
        self.assertAlmostEqual(config().square_size_m, 0.025)
        self.assertAlmostEqual(config(square_size=2.5, square_unit="cm").square_size_m, 0.025)
        self.assertAlmostEqual(config(square_size=0.025, square_unit="m").square_size_m, 0.025)

    def test_invalid_configs_are_rejected(self):
        cases = {
            "pattern too small": dict(pattern_size=(2, 6)),
            "pattern not ints": dict(pattern_size=(9.0, 6)),
            "pattern wrong length": dict(pattern_size=(9, 6, 1)),
            "pattern bool": dict(pattern_size=(True, 6)),
            "zero square": dict(square_size=0),
            "negative square": dict(square_size=-1),
            "nan square": dict(square_size=float("nan")),
            "inf square": dict(square_size=float("inf")),
            "string square": dict(square_size="25"),
            "unknown unit": dict(square_unit="inch"),
            "unknown model": dict(model="pinhole"),
            "too few min views": dict(min_views=2),
            "bool min views": dict(min_views=True),
            "bad detector": dict(detector="harris"),
            "bad expected size": dict(expected_image_size=(640, 0)),
            "bad rms": dict(max_rms_px=0),
            "coverage > 1": dict(min_coverage_fraction=1.5),
        }
        for label, kw in cases.items():
            with self.subTest(label), self.assertRaises(InvalidCalibrationInputError):
                config(**kw)

    def test_units_are_mandatory(self):
        with self.assertRaises(TypeError):
            IntrinsicCalibrationConfig(pattern_size=PATTERN, square_size=25)   # type: ignore[call-arg]

    def test_object_points_layout(self):
        obj = chessboard_object_points((4, 3), 0.1)
        self.assertEqual(obj.shape, (12, 3))
        np.testing.assert_allclose(obj[1], [0.1, 0, 0])      # next corner along a row
        np.testing.assert_allclose(obj[4], [0, 0.1, 0])      # first corner of the second row
        np.testing.assert_allclose(obj[11], [0.3, 0.2, 0])


class IntrinsicParametersTests(unittest.TestCase):
    def test_rejects_malformed_parameters(self):
        K = STD_CAMERA.camera_matrix()
        D = np.zeros(5)
        bad = {
            "size": (0, 480, K, D, "standard5"),
            "model": (640, 480, K, D, "mystery"),
            "dist length": (640, 480, K, np.zeros(4), "standard5"),
            "fisheye length": (640, 480, K, np.zeros(5), "fisheye"),
            "nan K": (640, 480, np.where(np.eye(3) > 0, np.nan, 0), D, "standard5"),
            "negative focal": (640, 480, K * np.array([[-1, 1, 1], [1, 1, 1], [1, 1, 1]]), D, "standard5"),
            "principal point outside": (640, 480, np.array([[500, 0, 9000], [0, 500, 240], [0, 0, 1.0]]), D, "standard5"),
            "skewed": (640, 480, np.array([[500, 5, 320], [0, 500, 240], [0, 0, 1.0]]), D, "standard5"),
            "bad last row": (640, 480, np.array([[500, 0, 320], [0, 500, 240], [0, 0, 2.0]]), D, "standard5"),
            "nan dist": (640, 480, K, np.array([np.nan, 0, 0, 0, 0]), "standard5"),
            "huge dist": (640, 480, K, np.array([1e9, 0, 0, 0, 0]), "standard5"),
        }
        for label, args in bad.items():
            with self.subTest(label), self.assertRaises(InvalidCalibrationInputError):
                IntrinsicParameters(*args)

    def test_dict_round_trip_through_json(self):
        calib = IntrinsicCalibrator(config())
        res = calib.calibrate_from_detections(detections_for(STD_CAMERA), (640, 480))
        text = json.dumps(res.intrinsics.to_dict(), allow_nan=False)
        back = IntrinsicParameters.from_dict(json.loads(text))
        np.testing.assert_allclose(back.camera_matrix, res.intrinsics.camera_matrix)
        np.testing.assert_allclose(back.dist_coeffs, res.intrinsics.dist_coeffs)
        self.assertEqual(back.model, "standard5")
        self.assertEqual(back.pattern_size, PATTERN)
        self.assertAlmostEqual(back.square_size_m, 0.025)
        self.assertEqual(back.quality.status, res.quality.status)

    def test_from_dict_reports_missing_fields(self):
        with self.assertRaisesRegex(InvalidCalibrationInputError, "missing fields"):
            IntrinsicParameters.from_dict({"image_width": 640})

    def test_arrays_are_read_only(self):
        p = STD_CAMERA.intrinsics()
        with self.assertRaises(ValueError):
            p.camera_matrix[0, 0] = 1.0

    def test_point_round_trip_for_every_model(self):
        cameras = {"standard5": STD_CAMERA, "fisheye": FISH_CAMERA,
                   "radial2": SyntheticCamera(dist=(-0.25, 0.08, 0, 0, 0), model="radial2"),
                   "rational8": SyntheticCamera(dist=(-0.2, 0.05, 0, 0, 0, 0.1, 0.02, 0.0), model="rational8")}
        us, vs = np.meshgrid(np.linspace(0, 639, 17), np.linspace(0, 479, 13))
        raw = np.column_stack([us.ravel(), vs.ravel()])
        for model, cam in cameras.items():
            with self.subTest(model):
                intr = cam.intrinsics()
                Knew = cam.camera_matrix() * np.array([[0.8, 1, 1], [1, 0.8, 1], [1, 1, 1]])
                back = intr.distort_points(intr.undistort_points(raw, Knew), Knew)
                self.assertLess(np.abs(back - raw).max(), 1e-6)

    def test_nonfinite_points_become_nan_not_garbage(self):
        out = STD_CAMERA.intrinsics().undistort_points([[10.0, 10.0], [np.nan, 5.0]])
        self.assertTrue(np.isfinite(out[0]).all())
        self.assertTrue(np.isnan(out[1]).all())

    def test_scaled_to_is_consistent_with_pixel_centre_convention(self):
        intr = STD_CAMERA.intrinsics()
        half = intr.scaled_to(320, 240)
        raw = np.array([[100.0, 80.0], [500.0, 400.0]])
        # An ideal ray seen at raw pixel p appears at s*(p+0.5)-0.5 in the half-size image.
        und_full = intr.undistort_points(raw, intr.camera_matrix)
        und_half = half.undistort_points(0.5 * (raw + 0.5) - 0.5, half.camera_matrix)
        np.testing.assert_allclose(0.5 * (und_full + 0.5) - 0.5, und_half, atol=1e-6)

    def test_scaled_to_rejects_aspect_change(self):
        with self.assertRaisesRegex(InvalidCalibrationInputError, "aspect ratio"):
            STD_CAMERA.intrinsics().scaled_to(640, 360)


class CalibrationFromDetectionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        cams = {"standard5": STD_CAMERA, "fisheye": FISH_CAMERA,
                "radial2": SyntheticCamera(dist=(-0.25, 0.08, 0, 0, 0), model="radial2"),
                "rational8": SyntheticCamera(dist=(-0.2, 0.05, 0, 0, 0, 0.1, 0.02, 0.0), model="rational8")}
        for model, cam in cams.items():
            # The 4-term fisheye model needs more views: with 16 it over-fits and the quality gate
            # (correctly) refuses the result because the distortion folds back near the corners.
            views = 30 if model == "fisheye" else 16
            cls.results[model] = (cam, IntrinsicCalibrator(config(model=model)).calibrate_from_detections(
                detections_for(cam, views, seed=4), (cam.width, cam.height)))

    def test_recovers_known_parameters_for_every_model(self):
        for model, (cam, res) in self.results.items():
            with self.subTest(model):
                K = res.intrinsics.camera_matrix
                self.assertAlmostEqual(K[0, 0] / cam.fx, 1.0, delta=0.02)
                self.assertAlmostEqual(K[1, 1] / cam.fy, 1.0, delta=0.02)
                self.assertAlmostEqual(K[0, 2], cam.cx, delta=3.0)
                self.assertAlmostEqual(K[1, 2], cam.cy, delta=3.0)
                self.assertEqual(res.intrinsics.dist_coeffs.shape, (DISTORTION_MODELS[model],))
                self.assertLess(res.intrinsics.rms_reprojection_px, 0.3)   # injected noise is 0.1 px per axis
                self.assertNotEqual(res.quality.status, "poor")

    def test_distortion_coefficients_recovered_for_standard_model(self):
        _, res = self.results["standard5"]
        np.testing.assert_allclose(res.intrinsics.dist_coeffs[:2], STD_CAMERA.dist[:2], atol=0.05)

    def test_radial2_leaves_unused_terms_at_zero(self):
        _, res = self.results["radial2"]
        D = res.intrinsics.dist_coeffs
        self.assertEqual((D[2], D[3], D[4]), (0.0, 0.0, 0.0))

    def test_reprojection_error_matches_independent_opencv_calculation(self):
        cam = STD_CAMERA
        dets = detections_for(cam, 12, seed=9)
        res = IntrinsicCalibrator(config()).calibrate_from_detections(dets, (640, 480))
        obj = chessboard_object_points(PATTERN, SQUARE_M).astype(np.float32)
        rms_cv, K, D, rv, tv = cv2.calibrateCamera([obj] * 12, [c.astype(np.float32).reshape(-1, 1, 2) for _, c in dets],
                                                  (640, 480), None, None)[:5]
        self.assertAlmostEqual(res.intrinsics.rms_reprojection_px, rms_cv, delta=1e-4)
        # per view: recompute straight from OpenCV projections
        for (_, c), r, t, mine in zip(dets, rv, tv, res.intrinsics.per_view_rms_px):
            proj = cv2.projectPoints(obj, r, t, K, D)[0].reshape(-1, 2)
            # OpenCV is fed float32 corners/object points, hence a 1e-4 px (not 1e-9) tolerance
            self.assertAlmostEqual(mine, math.sqrt(np.mean(np.sum((proj - c) ** 2, axis=1))), delta=1e-4)
        # overall RMS is consistent with the per-view figures
        self.assertAlmostEqual(res.intrinsics.rms_reprojection_px,
                               math.sqrt(np.mean(np.square(res.intrinsics.per_view_rms_px))), delta=1e-9)

    def test_quality_metrics_reported(self):
        _, res = self.results["standard5"]
        for key in ("rms_px", "orientation_spread_deg", "coverage_fraction", "max_roundtrip_error_px"):
            self.assertIn(key, res.quality.metrics)
        self.assertLess(res.quality.metrics["max_roundtrip_error_px"], 1e-6)
        self.assertIn("RMS", res.report())

    def test_too_few_views_is_an_error(self):
        with self.assertRaisesRegex(InsufficientCalibrationDataError, "min_views"):
            IntrinsicCalibrator(config(min_views=10)).calibrate_from_detections(detections_for(STD_CAMERA, 6), (640, 480))

    def test_wrong_corner_count_is_rejected(self):
        dets = detections_for(STD_CAMERA, 10)
        dets[3] = (dets[3][0], dets[3][1][:-1])
        with self.assertRaisesRegex(InvalidCalibrationInputError, "expected 54"):
            IntrinsicCalibrator(config()).calibrate_from_detections(dets, (640, 480))

    def test_nonfinite_corners_are_rejected(self):
        dets = detections_for(STD_CAMERA, 10)
        bad = dets[0][1].copy(); bad[5, 0] = np.nan
        dets[0] = (dets[0][0], bad)
        with self.assertRaises(InvalidCalibrationInputError):
            IntrinsicCalibrator(config()).calibrate_from_detections(dets, (640, 480))

    def test_image_size_must_match_expected(self):
        with self.assertRaises(InvalidCalibrationInputError):
            IntrinsicCalibrator(config(expected_image_size=(1280, 720))).calibrate_from_detections(
                detections_for(STD_CAMERA, 10), (640, 480))

    def test_similar_board_orientations_are_refused(self):
        # every view fronto-parallel: focal length is unobservable, so this must not pass as success
        dets = detections_for(STD_CAMERA, 12, seed=2, max_tilt_deg=2.0)
        with self.assertRaises(CalibrationQualityError) as ctx:
            IntrinsicCalibrator(config()).calibrate_from_detections(dets, (640, 480))
        self.assertIn("orientations too similar", str(ctx.exception))
        self.assertEqual(ctx.exception.result.quality.status, "poor")

    def test_poor_quality_can_be_inspected_when_enforcement_is_off(self):
        dets = detections_for(STD_CAMERA, 12, seed=2, max_tilt_deg=2.0)
        res = IntrinsicCalibrator(config(enforce_quality=False)).calibrate_from_detections(dets, (640, 480))
        self.assertEqual(res.quality.status, "poor")

    def test_insufficient_image_coverage_is_refused(self):
        poses = random_board_poses(STD_CAMERA, PATTERN, SQUARE_M, 12, seed=5, depth_range=(0.55, 0.75))
        dets = []
        for i, (R, t) in enumerate(poses):
            c = project_board_corners(STD_CAMERA, PATTERN, SQUARE_M, R, t)
            dets.append((f"v{i}", c * 0.3 + np.array([20.0, 15.0])))      # squeezed into one image corner
        with self.assertRaises(CalibrationQualityError) as ctx:
            IntrinsicCalibrator(config()).calibrate_from_detections(dets, (640, 480))
        self.assertTrue(any("cover only" in m for m in ctx.exception.result.quality.issues)
                        or any("RMS" in m for m in ctx.exception.result.quality.issues))

    def test_noisy_detections_exceed_rms_threshold(self):
        dets = detections_for(STD_CAMERA, 14, noise=2.5, seed=6)
        with self.assertRaises(CalibrationQualityError) as ctx:
            IntrinsicCalibrator(config(max_rms_px=1.0)).calibrate_from_detections(dets, (640, 480))
        self.assertIn("exceeds max_rms_px", str(ctx.exception))

    def test_thresholds_are_configurable(self):
        dets = detections_for(STD_CAMERA, 14, noise=2.5, seed=6)
        res = IntrinsicCalibrator(config(max_rms_px=50.0, min_coverage_fraction=0.0, min_orientation_spread_deg=1.0)
                                  ).calibrate_from_detections(dets, (640, 480))
        self.assertIn(res.quality.status, ("ok", "warning"))

    def test_non_invertible_distortion_is_detected(self):
        calib = IntrinsicCalibrator(config())
        K = STD_CAMERA.camera_matrix()
        sane = calib._invertibility_error(K, np.array([-0.28, 0.09, 0.001, -0.0008, -0.01]), (640, 480))
        # k1 = -1: r_d = r (1 - r^2) peaks at r = 0.577, inside this image (corner radius ~0.77), so it folds back
        folded = calib._invertibility_error(K, np.array([-1.0, 0.0, 0.0, 0.0, 0.0]), (640, 480))
        self.assertLess(sane, 1e-6)
        self.assertGreater(folded, 0.5)

    def test_same_parity_pattern_is_flagged(self):
        cfg = IntrinsicCalibrationConfig((7, 5), 25, "mm")
        cam = STD_CAMERA
        poses = random_board_poses(cam, (7, 5), SQUARE_M, 12, seed=1)
        dets = [(f"v{i}", project_board_corners(cam, (7, 5), SQUARE_M, R, t)) for i, (R, t) in enumerate(poses)]
        res = IntrinsicCalibrator(cfg).calibrate_from_detections(dets, (640, 480))
        self.assertTrue(any("same parity" in m for m in res.quality.issues))


class ImageWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        poses = random_board_poses(STD_CAMERA, PATTERN, SQUARE_M, 14, seed=21)
        cls.images = [(f"board_{i:02d}.png", render_chessboard(STD_CAMERA, PATTERN, SQUARE_M, R, t, seed=i)) for i, (R, t) in enumerate(poses)]
        cls.poses = poses

    def test_directory_workflow_end_to_end_with_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name, img in self.images:
                self.assertTrue(cv2.imwrite(str(folder / name), img))
            cv2.imwrite(str(folder / "blank.png"), np.full((480, 640, 3), 140, np.uint8))
            cv2.imwrite(str(folder / "other_size.png"), np.full((240, 320, 3), 140, np.uint8))
            (folder / "corrupt.png").write_bytes(b"not an image")
            (folder / "notes.txt").write_text("ignored")
            res = IntrinsicCalibrator(config()).calibrate_directory(folder)
        used = [v for v in res.views if v.used]
        rejected = {v.name: v.reason for v in res.views if not v.used}
        self.assertEqual(len(used), 14)
        self.assertIn("not detected", rejected["blank.png"])
        self.assertIn("differs", rejected["other_size.png"])
        self.assertIn("unreadable", rejected["corrupt.png"])
        self.assertNotIn("notes.txt", rejected)
        K = res.intrinsics.camera_matrix
        self.assertAlmostEqual(K[0, 0] / STD_CAMERA.fx, 1.0, delta=0.02)
        self.assertLess(res.intrinsics.rms_reprojection_px, 0.5)
        self.assertNotEqual(res.quality.status, "poor")

    def test_detector_variants_find_the_same_corners(self):
        R, t = self.poses[0]
        exact = project_board_corners(STD_CAMERA, PATTERN, SQUARE_M, R, t)
        gray = cv2.cvtColor(self.images[0][1], cv2.COLOR_BGR2GRAY)
        for detector in ("classic", "sb"):
            with self.subTest(detector):
                corners = IntrinsicCalibrator(config(detector=detector)).detect_corners(gray)
                self.assertIsNotNone(corners)
                err = np.linalg.norm(corners - exact, axis=1)
                if np.linalg.norm(corners[0] - exact[-1]) < np.linalg.norm(corners[0] - exact[0]):
                    err = np.linalg.norm(corners[::-1] - exact, axis=1)      # 'sb' may start from the other end
                self.assertLess(err.mean(), 0.4)

    def test_missing_directory_and_empty_directory(self):
        with self.assertRaises(InvalidCalibrationInputError):
            IntrinsicCalibrator(config()).calibrate_directory("definitely/not/a/dir")
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(InsufficientCalibrationDataError):
            IntrinsicCalibrator(config()).calibrate_directory(tmp)

    def test_no_corners_anywhere(self):
        blanks = [(f"b{i}", np.full((480, 640), 128, np.uint8)) for i in range(12)]
        with self.assertRaisesRegex(InsufficientCalibrationDataError, "no chessboard detected"):
            IntrinsicCalibrator(config()).calibrate_images(blanks)

    def test_wrong_pattern_dimensions_are_not_detected(self):
        wrong = IntrinsicCalibrator(config(pattern_size=(8, 6)))
        self.assertIsNone(wrong.detect_corners(cv2.cvtColor(self.images[0][1], cv2.COLOR_BGR2GRAY)))

    def test_detect_requires_grayscale_uint8(self):
        with self.assertRaises(InvalidCalibrationInputError):
            IntrinsicCalibrator(config()).detect_corners(self.images[0][1])


if __name__ == "__main__":
    unittest.main()
