import unittest
from dataclasses import replace

import cv2
import numpy as np

from camera_calibration import BEVProcessor, ProcessorConfig
from camera_calibration.test_bev import PROFILE
from camera_calibration.test_pipeline import packet
from .config import LaneConfig
from .preprocessing import Preprocessor


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = BEVProcessor(PROFILE, ProcessorConfig(max_frame_age_s=None)).process(packet())

    def frame(self, image, valid=None, support=None):
        full = np.full(image.shape[:2], 255, np.uint8)
        return replace(self.base, image=image, valid_mask=full if valid is None else valid,
                       support_mask=full if support is None else support)

    def test_white_yellow_and_non_marking_colors(self):
        image = np.full_like(self.base.image, 35)
        for x, color in ((40, (255, 255, 255)), (100, (0, 230, 230)), (160, (0, 0, 255))):
            cv2.line(image, (x, 10), (x, 480), color, 7)
        p = Preprocessor(LaneConfig()).process(self.frame(image))
        self.assertEqual(p.marking_mask[200, 40], 255)
        self.assertEqual(p.marking_mask[200, 100], 255)
        self.assertEqual(p.marking_mask[200, 160], 0)

    def test_invalid_and_unsupported_bright_regions_excluded(self):
        image = np.full_like(self.base.image, 255)
        valid = np.zeros(image.shape[:2], np.uint8)
        valid[20:-20, 20:-20] = 255
        support = np.zeros_like(valid)
        support[80:-80, 80:-80] = 255
        p = Preprocessor(LaneConfig()).process(self.frame(image, valid, support))
        self.assertFalse(np.any(p.marking_mask[support == 0]))
        self.assertFalse(np.any(p.edges[p.roi == 0]))
        self.assertFalse(np.any(p.marking_mask[80:82]))

    def test_local_contrast_recovers_dim_marking_under_shadow(self):
        image = np.full_like(self.base.image, 35)
        cv2.line(image, (100, 10), (100, 480), (120, 120, 120), 7)
        f = self.frame(image)
        fixed = Preprocessor(LaneConfig()).process(f)
        adaptive = Preprocessor(LaneConfig(preprocessing="local_contrast", white_min_value=90)).process(f)
        self.assertEqual(np.count_nonzero(fixed.marking_mask), 0)
        self.assertGreater(np.count_nonzero(adaptive.marking_mask), 1000)

    def test_opening_does_not_fill_gaps_and_inputs_are_unchanged(self):
        image = np.full_like(self.base.image, 35)
        cv2.line(image, (100, 20), (100, 200), (255, 255, 255), 7)
        cv2.line(image, (100, 230), (100, 480), (255, 255, 255), 7)
        before = image.copy()
        p = Preprocessor(LaneConfig(opening_size=3)).process(self.frame(image))
        self.assertEqual(p.marking_mask[215, 100], 0)
        np.testing.assert_array_equal(image, before)
        self.assertFalse(p.marking_mask.flags.writeable)

    def test_roi_cache_uses_shared_readonly_m02_masks(self):
        pre = Preprocessor(LaneConfig())
        self.assertIs(pre.roi(self.base), pre.roi(self.base))


if __name__ == "__main__":
    unittest.main()
