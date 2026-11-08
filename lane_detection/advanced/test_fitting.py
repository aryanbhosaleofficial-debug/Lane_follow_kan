import unittest
from dataclasses import replace

import cv2
import numpy as np

from .config import LaneConfig
from .extraction import extract
from .fitting import fit_boundary
from .test_extraction import mask_input


class FittingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = LaneConfig()

    def test_gentle_and_tight_curves_independent_truth(self):
        v = np.arange(5, 235, 3, dtype=float)
        for a in (0.0008, .003):
            truth = 70 + a * (v - 120) ** 2
            mask = np.zeros((240, 320), np.uint8)
            cv2.polylines(mask, [np.rint(np.column_stack((truth, v))).astype(np.int32)], False, 255, 5)
            tracks = extract(mask_input(mask), self.cfg).tracks
            fit = fit_boundary(max(tracks, key=len), self.cfg, 240, 0)
            self.assertIsNotNone(fit)
            self.assertEqual(fit.diagnostics.degree, 2)
            self.assertLess(np.max(np.abs(fit.polynomial.evaluate(v) - truth)), 1.2)
            self.assertLess(fit.diagnostics.residual_rms_px, 1)
            self.assertLess(fit.diagnostics.condition_number, 5)

    def test_straight_fit_falls_back_to_linear(self):
        v = np.arange(5, 235, 3, dtype=float)
        fit = fit_boundary(np.column_stack((70 + .08 * v, v)), self.cfg, 240, 0)
        self.assertEqual(fit.diagnostics.degree, 1)
        self.assertLess(abs(fit.polynomial.coefficients[0]), 1e-12)

    def test_robust_fit_removes_known_outliers_deterministically(self):
        v = np.arange(5, 235, 3, dtype=float)
        truth = 70 + .0015 * (v - 120) ** 2
        u = truth.copy()
        u[::7] += 30
        p = np.column_stack((u, v))
        fit = fit_boundary(p, self.cfg, 240, 0)
        self.assertEqual(fit, fit_boundary(p, self.cfg, 240, 0))
        self.assertLess(np.max(np.abs(fit.polynomial.evaluate(v) - truth)), .01)
        self.assertIn("outliers_removed", fit.reason_codes)
        self.assertEqual(fit.diagnostics.inlier_points + fit.diagnostics.endpoint_trimmed_points,
                         len(v) - len(v[::7]))
        self.assertEqual(fit.diagnostics.endpoint_trimmed_points, 2)

    def test_insufficient_degenerate_and_nonfinite_evidence(self):
        for p in (np.zeros((2, 2)), np.ones((40, 2)), np.column_stack((np.ones(20), np.arange(20)))):
            self.assertIsNone(fit_boundary(p, self.cfg, 240, 0))
        with self.assertRaises(ValueError):
            fit_boundary(np.array([[np.nan, 1]]), self.cfg, 240, 0)

    def test_implausible_curvature_is_rejected(self):
        v = np.arange(10, 230, 3, dtype=float)
        p = np.column_stack((70 + .003 * (v - 120) ** 2, v))
        self.assertIsNone(fit_boundary(p, replace(self.cfg, max_curvature_px_inv=.001), 240, 0))

    def test_evidence_loss_reduces_confidence_and_far_coverage(self):
        v = np.arange(5, 235, 3, dtype=float)
        p = np.column_stack((np.full_like(v, 70), v))
        full = fit_boundary(p, self.cfg, 240, 0)
        near = fit_boundary(p[v >= 100], self.cfg, 240, 0)
        self.assertLess(near.confidence, full.confidence)
        self.assertLess(near.diagnostics.far_coverage, full.diagnostics.far_coverage)
        self.assertLess(near.diagnostics.far_confidence, full.diagnostics.far_confidence)
        self.assertGreater(near.diagnostics.near_confidence, near.diagnostics.far_confidence)
        self.assertGreaterEqual(near.polynomial.v_min, 100)


if __name__ == "__main__":
    unittest.main()
