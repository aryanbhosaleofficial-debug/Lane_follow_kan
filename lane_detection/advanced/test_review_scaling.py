from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from camera_calibration import BEVConfig
from . import LaneConfig, LaneDetector
from .adversarial import arc_scene, circle_u, clutter_scene
from .synthetic import bev_fixture, scene


class BudgetTests(unittest.TestCase):
    def test_operation_budget_is_explicitly_uncertain(self):
        with patch("lane_detection.advanced.pipeline.fit_boundary", side_effect=AssertionError("budget exhausted")):
            result = LaneDetector(LaneConfig(max_association_comparisons=1)).process(bev_fixture(scene()))
        self.assertEqual(result.validity, "uncertain")
        self.assertIn("processing_budget_exceeded", result.reason_codes)
        self.assertFalse(result.boundaries)

    def test_capacity_exhaustion_does_not_fit_or_pair_partial_results(self):
        frame = bev_fixture(clutter_scene(320, 240))
        cfg = LaneConfig(max_tracks=6)
        with patch("lane_detection.advanced.pipeline.fit_boundary", side_effect=AssertionError("incomplete frame must not be fitted")):
            result = LaneDetector(cfg).process(frame)
        self.assertFalse(result.valid)
        self.assertIsNone(result.selected_pair)
        self.assertIn("candidate_capacity_exceeded", result.reason_codes)


class ScalingTests(unittest.TestCase):
    def test_equivalent_constant_normal_width_arcs_and_scale_binding(self):
        for scale in (1, 2):
            grid = BEVConfig(320 * scale, 240 * scale, .004 / scale, 160 * scale, 239 * scale)
            cfg = LaneConfig.for_bev(grid, (.38, .42))
            # arc_scene's reference row is v=120; shift span/centre consistently
            # by constructing at base resolution, then nearest-neighbour scaling.
            import cv2
            base, circle = arc_scene(700, 100, .2, (35, 205))
            image = cv2.resize(base, (320 * scale, 240 * scale), interpolation=cv2.INTER_NEAREST)
            frame = bev_fixture(image)
            from camera_calibration import CameraGeometry
            profile = frame.geometry.profile.with_bev(grid)
            frame = replace(frame, bev=grid, geometry=CameraGeometry(profile), profile_fingerprint=profile.fingerprint())
            result = LaneDetector(cfg).process(frame)
            self.assertTrue(result.valid, result.reason_codes)
            self.assertAlmostEqual(result.selected_pair.width_mean_px / scale, 100, delta=1)
            wrong = replace(cfg, configuration_meters_per_pixel=.005)
            with self.assertRaises(ValueError):
                LaneDetector(wrong).process(frame)
            strict = LaneConfig.for_bev(grid, (.38, .42), require_validated_metric=True)
            self.assertFalse(LaneDetector(strict).process(frame).valid)

    def test_equivalent_physical_scene_on_multiple_bev_grids(self):
        for scale in (1, 2):
            grid = BEVConfig(320 * scale, 240 * scale, .004 / scale, 160 * scale, 239 * scale)
            cfg = LaneConfig.for_bev(grid, (.38, .42), reference_meters_per_pixel=.004)
            image = scene(((0, 0, 70 * scale), (0, 0, 170 * scale)), width=320 * scale, height=240 * scale)
            frame = bev_fixture(image)
            profile = frame.geometry.profile.with_bev(grid)
            from camera_calibration import CameraGeometry
            frame = replace(frame, bev=grid, geometry=CameraGeometry(profile), profile_fingerprint=profile.fingerprint())
            result = LaneDetector(cfg).process(frame)
            self.assertTrue(result.valid, result.reason_codes)
            self.assertFalse(result.metric_validated)
            self.assertIsNone(result.selected_pair.width_mean_m)
            self.assertIn("physical_configuration_unverified", result.reason_codes)
            self.assertAlmostEqual(result.selected_pair.width_mean_px / scale, 100, delta=.5)
            self.assertEqual(cfg.row_step, 3 * scale)
            self.assertEqual(cfg.max_marking_width_px, 20 * scale)

    def test_invalid_scale_and_physical_width(self):
        grid = BEVConfig(320, 240, .004, 160, 239)
        for width, resolution in (((.5, .4), .004), ((.4, .5), 0), ((.4, .5), float("nan"))):
            with self.assertRaises(ValueError):
                LaneConfig.for_bev(grid, width, reference_meters_per_pixel=resolution)
