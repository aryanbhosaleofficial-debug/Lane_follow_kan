import unittest
from dataclasses import replace

import numpy as np

from .association import associate
from .config import LaneConfig
from .fitting import fit_boundary


class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = LaneConfig(min_lane_width_px=85, max_lane_width_px=115)
        self.roi = np.full((240, 320), 255, np.uint8)
        self.v = np.arange(5, 235, 3, dtype=float)

    def boundary(self, coeff, i):
        p = np.column_stack((np.polyval(coeff, self.v), self.v))
        return fit_boundary(p, self.cfg, 240, i)

    def test_unique_pair_identifies_sides_without_image_half(self):
        boundaries = (self.boundary((0, 0, 180), 1), self.boundary((0, 0, 280), 2))
        result = associate(boundaries, self.roi, self.cfg)
        self.assertFalse(result.ambiguous)
        self.assertEqual(len(result.pairs), 1)
        pair = result.pairs[0]
        self.assertEqual((pair.left_id, pair.right_id), (1, 2))
        self.assertAlmostEqual(float(pair.centreline.evaluate(120)), 230)
        self.assertIsNone(pair.width_mean_m)

    def test_multi_lane_keeps_adjacent_pairs_without_selection(self):
        boundaries = tuple(self.boundary((0, 0, u), i) for i, u in enumerate((50, 150, 250)))
        result = associate(boundaries, self.roi, self.cfg)
        self.assertTrue(result.ambiguous)
        self.assertEqual(len(result.pairs), 2)
        self.assertIn("multiple_compatible_pairs", result.reason_codes)

    def test_crossing_rejected_over_whole_domain(self):
        boundaries = (self.boundary((0, .55, 50), 0), self.boundary((0, -.55, 180), 1))
        result = associate(boundaries, self.roi, self.cfg)
        self.assertTrue(result.ambiguous)
        self.assertFalse(result.pairs)
        self.assertIn("boundary_intersection", result.reason_codes)

    def test_width_and_width_consistency(self):
        left = self.boundary((0, 0, 50), 0)
        for right, cfg, reason in ((self.boundary((0, 0, 80), 1), self.cfg, "lane_width_out_of_bounds"),
                                   (self.boundary((0, .2, 130), 1), replace(self.cfg, min_lane_width_px=60, max_lane_width_px=160), "lane_width_inconsistent")):
            result = associate((left, right), self.roi, cfg)
            self.assertFalse(result.pairs)
            self.assertIn(reason, result.reason_codes)

    def test_internal_quadratic_width_extremum_is_checked(self):
        left = self.boundary((0, 0, 50), 0)
        # End widths ~100 but middle width 75: outside configured minimum.
        right = self.boundary((.002, -.48, 153.8), 1)
        result = associate((left, right), self.roi, self.cfg)
        self.assertFalse(result.pairs)
        self.assertIn("lane_width_out_of_bounds", result.reason_codes)

    def test_roi_hole_invalidates_supported_pair(self):
        boundaries = (self.boundary((0, 0, 50), 0), self.boundary((0, 0, 150), 1))
        self.roi[100:130, 95:105] = 0
        self.assertFalse(associate(boundaries, self.roi, self.cfg).pairs)

    def test_metric_constraints_require_validated_scale(self):
        boundaries = (self.boundary((0, 0, 50), 0), self.boundary((0, 0, 150), 1))
        cfg = replace(self.cfg, lane_width_m=(.38, .42))
        self.assertIn("metric_constraints_unavailable", associate(boundaries, self.roi, cfg).reason_codes)
        pair = associate(boundaries, self.roi, cfg, .004).pairs[0]
        self.assertAlmostEqual(pair.width_mean_m, .4)


if __name__ == "__main__":
    unittest.main()
