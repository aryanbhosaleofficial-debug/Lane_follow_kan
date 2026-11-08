import unittest
from dataclasses import replace

import numpy as np

from camera_calibration import BEVProcessor, ProcessorConfig
from camera_calibration.test_bev import PROFILE
from camera_calibration.test_pipeline import packet
from .config import LaneConfig
from .models import Polynomial
from .validation import validate_frame


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = BEVProcessor(PROFILE, ProcessorConfig(max_frame_age_s=None)).process(packet())

    def test_configuration_rejects_invalid_ranges(self):
        for kw in ({"canny_low": 120}, {"blur_size": 2}, {"row_step": 0},
                   {"min_points": 2}, {"min_coverage": 1}, {"local_contrast": float("nan")},
                   {"min_lane_width_px": 300}, {"lane_width_m": (1, .5)},
                   {"white_min_value": 256}, {"clahe": 1}, {"max_gap_px": True},
                   {"single_boundary_role": "middle"}, {"nominal_lane_width_px": 2},
                   {"preprocessing": []}, {"single_boundary_role": []}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                LaneConfig(**kw)

    def test_real_m02_frame_contract(self):
        validate_frame(self.frame)
        self.assertEqual(self.frame.key, ("sess-A", 1))

    def test_bad_inputs(self):
        for f in (None, self.frame.image):
            with self.assertRaises(TypeError):
                validate_frame(f)
        for f in (replace(self.frame, image=np.zeros((10, 10, 3), np.uint8)),
                  replace(self.frame, valid_mask=self.frame.valid_mask.astype(float)),
                  replace(self.frame, pixel_format="RGB"), replace(self.frame, source_frame_id=-1),
                  replace(self.frame, profile_fingerprint="wrong"), replace(self.frame, geometry=None),
                  replace(self.frame, bev=None)):
            with self.assertRaises(ValueError):
                validate_frame(f)

    def test_polynomial_equation(self):
        np.testing.assert_allclose(Polynomial((2, 3, 4), 0, 5).evaluate([0, 1, 2]), [4, 9, 18])


if __name__ == "__main__":
    unittest.main()
