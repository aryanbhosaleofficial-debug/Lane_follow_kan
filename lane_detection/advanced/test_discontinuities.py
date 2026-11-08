import unittest
from dataclasses import replace

import cv2
import numpy as np

from camera_calibration import BEVProcessor, ProcessorConfig
from camera_calibration.test_bev import PROFILE
from camera_calibration.test_pipeline import packet
from .config import LaneConfig
from .extraction import extract
from .fitting import fit_boundary
from .inference import infer_missing_boundary
from .test_extraction import mask_input


class DiscontinuityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = LaneConfig()

    def fitted(self, mask, cfg=None):
        cfg = cfg or self.cfg
        ext = extract(mask_input(mask), cfg)
        return [f for i, pts in enumerate(ext.tracks) if (f := fit_boundary(pts, cfg, len(mask), i)) is not None]

    def test_curved_dashes_associated_without_filling_observed_mask(self):
        mask = np.zeros((240, 320), np.uint8)
        for start in (8, 68, 128, 188):
            v = np.arange(start, min(start + 32, 233))
            u = 70 + .001 * (v - 120) ** 2
            cv2.polylines(mask, [np.rint(np.column_stack((u, v))).astype(np.int32)], False, 255, 5)
        original = mask.copy()
        fits = self.fitted(mask)
        self.assertEqual(len(fits), 1)
        fit = fits[0]
        self.assertEqual(len(fit.observed_intervals), 4)
        self.assertEqual(len(fit.unsupported_intervals), 3)
        self.assertLess(fit.confidence, .85)
        v = np.linspace(12, 215, 70)
        self.assertLess(np.max(np.abs(fit.polynomial.evaluate(v) - (70 + .001 * (v - 120) ** 2))), 1.5)
        np.testing.assert_array_equal(mask, original)
        self.assertEqual(mask[50, 75], 0)

    def test_short_broken_gap_is_explicit(self):
        mask = np.zeros((240, 320), np.uint8)
        cv2.line(mask, (70, 5), (70, 235), 255, 5)
        mask[100:135] = 0
        fit = self.fitted(mask)[0]
        self.assertEqual(len(fit.unsupported_intervals), 1)
        self.assertGreater(fit.diagnostics.max_gap_px, 30)
        self.assertIn("unsupported_gaps", fit.reason_codes)

    def test_adjacent_staggered_dashes_keep_distinct_curves(self):
        mask = np.zeros((240, 320), np.uint8)
        for x, offset in ((60, 0), (160, 20), (260, 0)):
            for start in range(5 + offset, 235, 60):
                cv2.line(mask, (x, start), (x, min(start + 30, 235)), 255, 5)
        fits = self.fitted(mask)
        self.assertEqual(len(fits), 3)
        for truth in (60, 160, 260):
            self.assertLess(min(abs(float(f.polynomial.evaluate(120)) - truth) for f in fits), 1)

    def test_long_gap_is_not_bridged(self):
        mask = np.zeros((240, 320), np.uint8)
        cv2.line(mask, (70, 5), (70, 65), 255, 5)
        cv2.line(mask, (70, 170), (70, 235), 255, 5)
        self.assertFalse(any(f.polynomial.v_min < 70 and f.polynomial.v_max > 160 for f in self.fitted(mask)))

    def test_inference_requires_prior_role_and_metric_trust(self):
        mask = np.zeros((240, 320), np.uint8)
        cv2.line(mask, (70, 5), (70, 235), 255, 5)
        observed = self.fitted(mask)[0]
        frame = BEVProcessor(PROFILE, ProcessorConfig(max_frame_age_s=None)).process(packet())
        self.assertIsNone(infer_missing_boundary(observed, self.cfg, frame))
        self.assertIsNone(infer_missing_boundary(observed, replace(self.cfg, nominal_lane_width_px=100), frame))
        cfg = replace(self.cfg, nominal_lane_width_px=100, single_boundary_role="left")
        inferred = infer_missing_boundary(observed, cfg, frame)
        self.assertFalse(inferred.observed)
        self.assertEqual(inferred.observed_intervals, ())
        self.assertLess(inferred.confidence, observed.confidence)
        self.assertAlmostEqual(float(inferred.polynomial.evaluate(120)), 170)
        metric_cfg = replace(self.cfg, nominal_lane_width_m=.5, single_boundary_role="left")
        self.assertIsNone(infer_missing_boundary(observed, metric_cfg, frame))


if __name__ == "__main__":
    unittest.main()
