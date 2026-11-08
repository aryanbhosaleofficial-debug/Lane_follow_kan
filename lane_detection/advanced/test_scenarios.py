"""Independent ground-truth pixel geometry; passing is not real-track evidence."""
import unittest
from dataclasses import replace

import cv2
import numpy as np

from . import LaneConfig, LaneDetector
from .synthetic import bev_fixture, scene


class SyntheticScenarioTests(unittest.TestCase):
    def assert_geometry(self, result, left, right, tolerance=2.0):
        self.assertTrue(result.valid, result.reason_codes)
        pair = result.selected_pair
        by_id = {b.candidate_id: b for b in result.boundaries}
        rows = np.linspace(pair.centreline.v_min, pair.centreline.v_max, 100)
        for identifier, truth in ((pair.left_id, left), (pair.right_id, right)):
            b = by_id[identifier]
            self.assertTrue(b.observed)
            self.assertLess(np.max(np.abs(b.polynomial.evaluate(rows) - np.polyval(truth, rows))), tolerance)
            self.assertLess(b.diagnostics.residual_rms_px, 1.5)
        centre_truth = (np.polyval(left, rows) + np.polyval(right, rows)) / 2
        self.assertLess(np.max(np.abs(pair.centreline.evaluate(rows) - centre_truth)), tolerance)

    def test_solid_gentle_tight_vertical_and_near_vertical_geometry(self):
        for a, b in ((0, 0), (.0008, -.192), (.003, -.72), (0, .08)):
            c = a * 120 ** 2
            left, right = (a, b, 70 + c), (a, b, 170 + c)
            with self.subTest(a=a, b=b):
                self.assert_geometry(LaneDetector().process(bev_fixture(scene((left, right)))), left, right)

    def test_dashed_and_broken_geometry_and_lower_confidence(self):
        left, right = (.001, -.24, 84.4), (.001, -.24, 184.4)
        solid = LaneDetector().process(bev_fixture(scene((left, right))))
        for options in ({"dashed": True}, {"broken": True}):
            with self.subTest(options=options):
                result = LaneDetector().process(bev_fixture(scene((left, right), **options)))
                self.assert_geometry(result, left, right)
                self.assertLess(result.confidence, solid.confidence)
                self.assertTrue(all(b.unsupported_intervals for b in result.boundaries))

    def test_shadow_local_contrast_improves_evidence(self):
        image = scene(shadow=True)
        fixed = LaneDetector().process(bev_fixture(image))
        adaptive = LaneDetector(LaneConfig(preprocessing="local_contrast", white_min_value=90)).process(bev_fixture(image))
        self.assert_geometry(adaptive, (0, 0, 70), (0, 0, 170))
        fixed_support = sum(b.diagnostics.coverage for b in fixed.boundaries)
        adaptive_support = sum(b.diagnostics.coverage for b in adaptive.boundaries)
        self.assertGreater(adaptive_support, fixed_support + .2)
        self.assertGreater(adaptive.confidence, fixed.confidence)

    def test_noise_and_spurious_fragments(self):
        image = scene(noise=True)
        result = LaneDetector().process(bev_fixture(image))
        self.assert_geometry(result, (0, 0, 70), (0, 0, 170), tolerance=2.5)
        self.assertEqual(len(result.boundaries), 2)
        empty = scene((), noise=True)
        self.assertFalse(LaneDetector().process(bev_fixture(empty)).valid)

    def test_combined_curved_dashed_noisy_markings(self):
        for scale in (1, 2):
            left, right = (.001 / scale, -.24, 84.4 * scale), (.001 / scale, -.24, 184.4 * scale)
            image = scene((left, right), width=320 * scale, height=240 * scale, dashed=True, noise=True)
            cfg = LaneConfig(min_lane_width_px=45 * scale, max_lane_width_px=260 * scale)
            with self.subTest(scale=scale):
                self.assert_geometry(LaneDetector(cfg).process(bev_fixture(image)), left, right, tolerance=2.5)

    def test_invalid_borders_and_calibration_support(self):
        frame = bev_fixture(scene())
        image = frame.image.copy()
        image[:, :25] = 255
        image[:, -25:] = 255
        mask = np.full((240, 320), 255, np.uint8)
        mask[:, :30] = mask[:, -30:] = 0
        support = mask.copy()
        support[:, 250:] = 0
        frame = replace(frame, image=image, valid_mask=mask, support_mask=support)
        detector = LaneDetector()
        result = detector.process(frame, debug=True)
        self.assert_geometry(result, (0, 0, 70), (0, 0, 170))
        self.assertFalse(np.any(detector.last_debug.candidate_mask[support == 0]))
        support[:, 160:180] = 0
        result = LaneDetector().process(replace(frame, support_mask=support))
        self.assertFalse(result.valid)
        self.assertIsNone(result.selected_pair)

    def test_multiple_lanes_preserve_candidates_without_centreline(self):
        result = LaneDetector().process(bev_fixture(scene(((0, 0, 50), (0, 0, 150), (0, 0, 250)))))
        self.assertEqual(len(result.boundaries), 3)
        self.assertEqual(len(result.compatible_pairs), 2)
        self.assertIn("multiple_compatible_pairs", result.reason_codes)
        self.assertFalse(result.valid)
        self.assertIsNone(result.selected_pair)

    def test_junction_fork_rejects_lane_selection(self):
        image = scene()
        cv2.line(image, (170, 220), (290, 10), (255, 255, 255), 5)
        result = LaneDetector().process(bev_fixture(image))
        self.assertFalse(result.valid, result)
        self.assertIsNone(result.selected_pair)

    def test_insufficient_fit_evidence_has_no_lane(self):
        image = np.full((240, 320, 3), 35, np.uint8)
        for x in (70, 170):
            cv2.line(image, (x, 190), (x, 230), (255, 255, 255), 5)
        result = LaneDetector().process(bev_fixture(image))
        self.assertEqual(result.validity, "invalid")
        self.assertFalse(result.boundaries)
        self.assertIsNone(result.selected_pair)

    def test_bad_size_masks_dtype_and_raw_input(self):
        f = bev_fixture(scene())
        for frame in (replace(f, image=np.zeros((241, 320, 3), np.uint8)),
                      replace(f, image=f.image.astype(float)), replace(f, support_mask=np.zeros((10, 10), np.uint8)),
                      replace(f, support_mask=np.full((240, 320), 1, np.uint8)),
                      replace(f, valid_mask=np.zeros((240, 320), np.uint8))):
            with self.subTest(shape=getattr(frame.image, "shape", None)), self.assertRaises(ValueError):
                LaneDetector().process(frame)
        with self.assertRaises(TypeError):
            LaneDetector().process(f.image)

    def test_resolution_640_480_geometry(self):
        # Width bounds explicitly scaled for this different pixel grid.
        left, right = (.0005, -.24, 168.8), (.0005, -.24, 368.8)
        cfg = LaneConfig(min_lane_width_px=170, max_lane_width_px=230)
        result = LaneDetector(cfg).process(bev_fixture(scene((left, right), width=640, height=480)))
        self.assert_geometry(result, left, right)


if __name__ == "__main__":
    unittest.main()
