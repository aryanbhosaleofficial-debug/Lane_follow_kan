import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from .config import LaneConfig
from .extraction import extract, hough_segments
from .preprocessing import Preprocessed
from lane_detection.lane_detection import extract_lane_segments


def mask_input(mask):
    return Preprocessed(np.full_like(mask, 255), mask, cv2.Canny(mask, 40, 120))


class ExtractionTests(unittest.TestCase):
    def test_vertical_and_displaced_straight_boundaries(self):
        mask = np.zeros((240, 320), np.uint8)
        for u in (180, 280):  # both on right half: no lane-side assumption
            cv2.line(mask, (u, 10), (u, 230), 255, 5)
        result = extract(mask_input(mask), LaneConfig(use_hough=True))
        self.assertEqual(len(result.tracks), 2)
        for points, truth in zip(result.tracks, (180, 280)):
            self.assertGreater(len(points), 65)
            self.assertLess(np.max(np.abs(points[:, 0] - truth)), 1)
        self.assertGreater(len(result.hough_segments), 0)

    def test_three_candidates_are_not_split_by_image_half(self):
        mask = np.zeros((240, 320), np.uint8)
        for u in (50, 150, 250):
            cv2.line(mask, (u, 5), (u + 10, 235), 255, 5)
        result = extract(mask_input(mask), LaneConfig())
        self.assertEqual(len(result.tracks), 3)
        self.assertFalse(result.branch_detected)

    def test_hough_vertical_duplicate_and_horizontal_rejection(self):
        votes = np.array([[[50, 10, 50, 230]], [[51, 20, 51, 220]], [[20, 100, 280, 100]]])
        with patch("cv2.HoughLinesP", return_value=votes):
            kept = hough_segments(np.zeros((240, 320), np.uint8), LaneConfig())
        self.assertEqual(kept, ((50, 10, 50, 230),))
        legacy_args = SimpleNamespace(hough_threshold=12, min_line_length=12, max_line_gap=15)
        with patch("cv2.HoughLinesP", return_value=votes[:1]):
            self.assertEqual(extract_lane_segments(np.zeros((240, 320), np.uint8), legacy_args), ([], []))

    def test_horizontal_and_short_isolated_fragments_do_not_form_boundary(self):
        mask = np.zeros((240, 320), np.uint8)
        cv2.line(mask, (20, 120), (280, 120), 255, 5)
        cv2.line(mask, (40, 30), (40, 35), 255, 3)
        tracks = extract(mask_input(mask), LaneConfig()).tracks
        self.assertTrue(all(len(p) < LaneConfig().min_points for p in tracks))


if __name__ == "__main__":
    unittest.main()
