"""Basic tests for the saved-calibration IPM transformer."""

import tempfile
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from lane_detection.ipm_transform import IPMTransformer


class TestIPMTransformer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "calibration.json"
        self.config = {
            "source_image_width": 320,
            "source_image_height": 240,
            "source_points_px": [[80, 60], [240, 60], [300, 220], [20, 220]],
            "destination_width_px": 200,
            "destination_height_px": 150,
            "homography": cv2.getPerspectiveTransform(
                np.float32([[80, 60], [240, 60], [300, 220], [20, 220]]),
                np.float32([[0, 0], [199, 0], [199, 149], [0, 149]])
            ).tolist(),
        }
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.transformer = IPMTransformer(self.config_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_output_dimensions(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        bev = self.transformer.transform(frame)
        self.assertEqual(bev.shape, (150, 200, 3))

    def test_rejects_wrong_input_dimensions(self):
        wrong_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.transformer.transform(wrong_frame)

    def test_rejects_invalid_homography_shape(self):
        bad = dict(self.config)
        bad["homography"] = [[1, 0], [0, 1]]
        self.config_path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(ValueError):
            IPMTransformer(self.config_path)


if __name__ == "__main__":
    unittest.main()
