"""Review acceptance regressions, written against the uncorrected implementation."""
from dataclasses import replace
import unittest

import numpy as np

from . import LaneConfig, LaneDetector
from .adversarial import arc_scene, circle_u, edge_scene
from .synthetic import bev_fixture, scene


class EdgeExitTests(unittest.TestCase):
    def test_roi_hole_outside_shared_domain_does_not_reject_pair(self):
        image = scene()
        image[:120, 160:180] = 35
        frame = bev_fixture(image)
        support = frame.support_mask.copy()
        support[40:60, 45:80] = 0
        # The unused far portion of the left candidate crosses an unsupported
        # calibration patch; both boundaries are supported on the near pair.
        result = LaneDetector().process(replace(frame, support_mask=support))
        self.assertTrue(result.valid, result.reason_codes)
        self.assertGreater(result.selected_pair.centreline.v_min, 115)

    def test_supported_left_and_right_edge_exits(self):
        for mirror in (False, True):
            for slope in (.15, .25, .4, .6, .8):
                for exit_row in (60, 100, 140):
                    with self.subTest(mirror=mirror, slope=slope, exit_row=exit_row):
                        image, truth = edge_scene(slope, exit_row, mirror)
                        result = LaneDetector().process(bev_fixture(image))
                        # At slope .15 and exit 140, a full 5px stroke clears the
                        # 2px ROI margin only near row 180: <65px common support.
                        if slope == .15 and exit_row == 140:
                            self.assertFalse(result.valid)
                            continue
                        self.assertTrue(result.valid, result.reason_codes)
                        pair = result.selected_pair
                        by_id = {b.candidate_id: b for b in result.boundaries}
                        rows = np.linspace(pair.centreline.v_min, pair.centreline.v_max, 60)
                        for identifier, abc in zip((pair.left_id, pair.right_id), truth):
                            self.assertLess(np.max(np.abs(by_id[identifier].polynomial.evaluate(rows) - np.polyval(abc, rows))), 1.5)

    def test_insufficient_common_span_stays_invalid(self):
        image, _ = edge_scene(.6, 205)
        self.assertFalse(LaneDetector().process(bev_fixture(image)).valid)


class NormalWidthTests(unittest.TestCase):
    def test_tight_arc_caps_never_create_inaccurate_valid_boundaries(self):
        successes = 0
        for radius, width, direction, span in ((220, 140, -.35, (65, 175)), (300, 100, .35, (35, 205)),
                                               (700, 60, -.35, (65, 175))):
            image, circle = arc_scene(radius, width, direction, span)
            cfg = LaneConfig(min_lane_width_px=width - 3, max_lane_width_px=width + 3,
                             max_residual_px=.75, inlier_threshold_px=1)
            result = LaneDetector(cfg).process(bev_fixture(image))
            if result.valid:
                successes += 1
                by_id = {b.candidate_id: b for b in result.boundaries}
                for identifier, offset in ((result.selected_pair.left_id, width / 2), (result.selected_pair.right_id, -width / 2)):
                    poly = by_id[identifier].polynomial
                    rows = np.linspace(poly.v_min, poly.v_max, 100)
                    self.assertLess(np.max(np.abs(poly.evaluate(rows) - circle_u(rows, circle, offset))), 1.5)
            else:
                self.assertIsNone(result.selected_pair)
            if radius == 700:
                self.assertTrue(result.valid, result.reason_codes)
        self.assertGreaterEqual(successes,1)

    def test_multiple_and_absent_normal_intersections_are_rejected(self):
        from .geometry import normal_geometry
        from .models import Polynomial
        cfg = LaneConfig()
        ambiguous = normal_geometry(Polynomial((0, 1, -200), 190, 210), Polynomial((.02, -5, 350), 30, 170), cfg)
        self.assertEqual(ambiguous.reason, "ambiguous_normal_intersection")
        absent = normal_geometry(Polynomial((0, 0, 70), 100, 200), Polynomial((0, 0, 170), 0, 50), cfg)
        self.assertEqual(absent.reason, "no_supported_normal_correspondence")

    def test_moderate_constant_normal_width_arcs(self):
        for radius in (400, 700, 1200):
            for width in (60, 100, 140):
                for orientation in (0.0, .35):
                    with self.subTest(radius=radius, width=width, orientation=orientation):
                        image, circle = arc_scene(radius, width, orientation, (35, 205))
                        cfg = LaneConfig(min_lane_width_px=width - 3, max_lane_width_px=width + 3)
                        result = LaneDetector(cfg).process(bev_fixture(image))
                        self.assertTrue(result.valid, result.reason_codes)
                        pair = result.selected_pair
                        self.assertLess(abs(pair.width_mean_px - width), 1.5)
                        rows = np.linspace(pair.centreline.v_min, pair.centreline.v_max, 60)
                        self.assertLess(np.max(np.abs(pair.centreline.evaluate(rows) - circle_u(rows, circle))), 1.5)


class CompanionTests(unittest.TestCase):
    def test_companions_do_not_silently_change_the_lane(self):
        for separation in (10, 14, 18, 24, 32, 45):
            for u in (70 + separation, 200 - separation, 70 - separation, 200 + separation):
                with self.subTest(separation=separation, u=u):
                    result = LaneDetector().process(bev_fixture(scene(tuple((0, 0, x) for x in sorted((70, 200, u))))))
                    self.assertFalse(result.valid, result)
                    self.assertIsNone(result.selected_pair)

    def test_clean_true_boundaries_remain_valid(self):
        result = LaneDetector().process(bev_fixture(scene(((0, 0, 70), (0, 0, 200)))))
        self.assertTrue(result.valid)
        self.assertAlmostEqual(float(result.selected_pair.centreline.evaluate(120)), 135, delta=.5)
