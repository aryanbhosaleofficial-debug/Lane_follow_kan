"""Tests for undistortion maps (SYNTHETIC data)."""
import unittest

import cv2
import numpy as np

from camera_calibration import undistort as undistort_mod
from camera_calibration.diagnostics import row_straightness
from camera_calibration.errors import InvalidCalibrationInputError, InvalidFrameError, ResolutionMismatchError
from camera_calibration.intrinsics import IntrinsicCalibrationConfig, IntrinsicCalibrator
from camera_calibration.synthetic import (SyntheticCamera, project_board_corners, random_board_poses, render_chessboard)
from camera_calibration.undistort import Undistorter, UndistortionSpec

PATTERN, SQUARE = (9, 6), 0.025
CAM = SyntheticCamera(dist=(-0.28, 0.09, 0.001, -0.0008, -0.01))
FISH = SyntheticCamera(fx=330.0, fy=330.0, dist=(0.04, -0.01, 0.002, -0.0005), model="fisheye")


def board_view(camera=CAM, seed=3, count=1):
    poses = random_board_poses(camera, PATTERN, SQUARE, count, seed=seed, max_tilt_deg=25)
    return poses[0], render_chessboard(camera, PATTERN, SQUARE, *poses[0], seed=seed)


class SpecTests(unittest.TestCase):
    def test_alpha_modes_produce_valid_matrices(self):
        intr = CAM.intrinsics()
        for alpha in (None, 0.0, 0.5, 1.0):
            with self.subTest(alpha=alpha):
                spec = UndistortionSpec.from_intrinsics(intr, alpha)
                self.assertEqual(spec.output_size, (640, 480))
                self.assertTrue(np.isfinite(spec.camera_matrix).all())
                self.assertEqual(spec.camera_matrix[2].tolist(), [0, 0, 1])
        np.testing.assert_allclose(UndistortionSpec.from_intrinsics(intr, None).camera_matrix, intr.camera_matrix)

    def test_alpha_controls_cropping_explicitly(self):
        intr = CAM.intrinsics()
        zoomed = Undistorter(intr, UndistortionSpec.from_intrinsics(intr, 0.0))
        full = Undistorter(intr, UndistortionSpec.from_intrinsics(intr, 1.0))
        self.assertGreater(zoomed.valid_fraction, 0.995)          # alpha=0: no black border
        self.assertLess(full.valid_fraction, 0.98)                # alpha=1: whole field of view, border visible
        fx0 = UndistortionSpec.from_intrinsics(intr, 0.0).camera_matrix[0, 0]
        fx1 = UndistortionSpec.from_intrinsics(intr, 1.0).camera_matrix[0, 0]
        self.assertGreater(fx0, fx1)                              # cropping zooms in

    def test_invalid_alpha_and_matrices_rejected(self):
        K = CAM.camera_matrix()
        for bad in (-0.1, 1.5, "0", True):
            with self.subTest(alpha=bad), self.assertRaises(InvalidCalibrationInputError):
                UndistortionSpec(K, bad, (640, 480))
        for bad_K in (np.full((3, 3), np.nan), np.eye(3) * 0, np.array([[500, 1, 320], [0, 500, 240], [0, 0, 1.0]])):
            with self.subTest(K=bad_K.tolist()), self.assertRaises(InvalidCalibrationInputError):
                UndistortionSpec(bad_K, 0.0, (640, 480))
        with self.assertRaises(InvalidCalibrationInputError):
            UndistortionSpec(K, 0.0, (0, 480))

    def test_dict_round_trip(self):
        spec = UndistortionSpec.from_intrinsics(CAM.intrinsics(), 0.3)
        back = UndistortionSpec.from_dict(spec.to_dict())
        np.testing.assert_allclose(back.camera_matrix, spec.camera_matrix)
        self.assertEqual((back.alpha, back.output_size), (0.3, (640, 480)))


class UndistortTests(unittest.TestCase):
    def test_output_has_input_dimensions_and_format(self):
        _, img = board_view()
        out = Undistorter(CAM.intrinsics()).undistort(img)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)
        gray = Undistorter(CAM.intrinsics()).undistort(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        self.assertEqual(gray.shape, img.shape[:2])

    def test_input_is_not_modified_even_when_read_only(self):
        _, img = board_view()
        img.flags.writeable = False
        before = img.copy()
        out = Undistorter(CAM.intrinsics()).undistort(img)
        np.testing.assert_array_equal(img, before)
        self.assertTrue(out.flags.writeable)
        self.assertFalse(np.shares_memory(out, img))

    def test_straight_lines_become_straight_via_points(self):
        for name, cam in (("standard5", CAM), ("fisheye", FISH)):
            with self.subTest(name):
                intr = cam.intrinsics()
                spec = UndistortionSpec.from_intrinsics(intr, 0.0)
                poses = random_board_poses(cam, PATTERN, SQUARE, 8, seed=2, max_tilt_deg=30)
                raw_rms, und_rms = [], []
                for pose in poses:
                    raw = project_board_corners(cam, PATTERN, SQUARE, *pose)
                    und = intr.undistort_points(raw, spec.camera_matrix)
                    raw_rms.append(row_straightness(raw, PATTERN)["rms_px"])
                    und_rms.append(row_straightness(und, PATTERN)["rms_px"])
                self.assertGreater(np.mean(raw_rms), 0.1)         # a small board still bows measurably in the raw points
                self.assertLess(max(und_rms), 1e-6)               # and removed exactly (up to float error) afterwards

    def test_straight_lines_become_straight_via_image(self):
        calib = IntrinsicCalibrator(IntrinsicCalibrationConfig(PATTERN, 25, "mm"))
        und = Undistorter(CAM.intrinsics())
        raw_vals, und_vals = [], []
        for seed in range(4):
            _, img = board_view(seed=seed + 7)
            raw = calib.detect_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            fixed = calib.detect_corners(cv2.cvtColor(und.undistort(img), cv2.COLOR_BGR2GRAY))
            if raw is None or fixed is None:
                self.fail(f"board not detected (raw={raw is not None}, undistorted={fixed is not None}) for seed {seed}")
            raw_vals.append(row_straightness(raw, PATTERN)["rms_px"])
            und_vals.append(row_straightness(fixed, PATTERN)["rms_px"])
        self.assertLess(np.mean(und_vals), 0.5 * np.mean(raw_vals))
        self.assertLess(np.mean(und_vals), 0.35)

    def test_image_path_agrees_with_point_path(self):
        # a corner found in the undistorted image must sit where undistort_points predicts it
        intr = CAM.intrinsics()
        spec = UndistortionSpec.from_intrinsics(intr, 0.0)
        pose, img = board_view(seed=4)
        calib = IntrinsicCalibrator(IntrinsicCalibrationConfig(PATTERN, 25, "mm"))
        found = calib.detect_corners(cv2.cvtColor(Undistorter(intr, spec).undistort(img), cv2.COLOR_BGR2GRAY))
        predicted = intr.undistort_points(project_board_corners(CAM, PATTERN, SQUARE, *pose), spec.camera_matrix)
        self.assertIsNotNone(found)
        self.assertLess(np.linalg.norm(found - predicted, axis=1).mean(), 0.5)

    def test_wrong_resolution_and_bad_images_are_rejected(self):
        und = Undistorter(CAM.intrinsics())
        with self.assertRaises(ResolutionMismatchError):
            und.undistort(np.zeros((240, 320, 3), np.uint8))
        for bad in (np.zeros((480, 640, 3), np.float32), np.zeros((480, 640, 4), np.uint8), np.zeros((0, 640, 3), np.uint8),
                    np.zeros((480, 640, 3, 1), np.uint8), [[0]], None):
            with self.subTest(type=type(bad).__name__), self.assertRaises(InvalidFrameError):
                und.undistort(bad)

    def test_spec_size_must_match_intrinsics(self):
        spec = UndistortionSpec(CAM.camera_matrix(), 0.0, (320, 240))
        with self.assertRaises(InvalidCalibrationInputError):
            Undistorter(CAM.intrinsics(), spec)

    def test_maps_are_cached_and_not_rebuilt_per_frame(self):
        undistort_mod._build_maps.cache_clear()
        intr = CAM.intrinsics()
        a, b = Undistorter(intr), Undistorter(intr)
        _, img = board_view()
        for _ in range(5):
            a.undistort(img); b.undistort(img)
        info = undistort_mod._build_maps.cache_info()
        self.assertEqual(info.misses, 1)       # one build serves both instances and all frames
        # a different calibration must not reuse the cached maps
        Undistorter(SyntheticCamera(dist=(-0.1, 0, 0, 0, 0)).intrinsics()).undistort(img)
        self.assertEqual(undistort_mod._build_maps.cache_info().misses, 2)

    def test_cached_maps_are_read_only(self):
        und = Undistorter(CAM.intrinsics())
        m1, m2, mask = und._ensure_maps()
        for arr in (m1, m2, mask):
            with self.assertRaises(ValueError):
                arr[0, 0] = 0

    def test_interpolation_options(self):
        _, img = board_view()
        outs = {m: Undistorter(CAM.intrinsics(), interpolation=m).undistort(img) for m in ("nearest", "linear", "cubic")}
        for m, o in outs.items():
            self.assertEqual(o.shape, img.shape)
        self.assertLess(np.abs(outs["linear"].astype(int) - outs["cubic"].astype(int)).mean(), 5)
        with self.assertRaises(InvalidCalibrationInputError):
            Undistorter(CAM.intrinsics(), interpolation="lanczos")

    def test_float_and_fixed_point_maps_agree(self):
        _, img = board_view()
        a = Undistorter(CAM.intrinsics(), fixed_point_maps=True).undistort(img).astype(int)
        b = Undistorter(CAM.intrinsics(), fixed_point_maps=False).undistort(img).astype(int)
        self.assertLess(np.abs(a - b).mean(), 1.0)

    def test_fisheye_image_path(self):
        out = Undistorter(FISH.intrinsics()).undistort(np.full((480, 640, 3), 90, np.uint8))
        self.assertEqual(out.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
