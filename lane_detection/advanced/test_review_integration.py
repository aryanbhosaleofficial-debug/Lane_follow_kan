"""Independent ground geometry rendered through real M01/M02, not pixel shifts."""
from dataclasses import replace
import unittest

import cv2
import numpy as np

from camera_calibration import BEVProcessor, ProcessorConfig
from camera_calibration.synthetic import RoadScene
from camera_calibration.test_bev import CAM, POSE, PROFILE, CURVE_IMG
from camera_calibration.test_pipeline import RoadCapture, packet
from camera_calibration.test_status import fixture_profile, report_for
from camera_input import CameraConfig, CameraManager
from . import LaneConfig, LaneDetector


def physical_scene(curved=False):
    ppm, x0, y0 = 700, -.5, -1.8
    texture = np.full((2520, 3150, 3), 30, np.uint8)
    cx, cy, radius, width = .2, 2.0, 2.0, .5
    for r, y in ((radius - width / 2, .25), (radius + width / 2, -.25)):
        if curved:
            theta = np.linspace(-np.pi / 2, 0, 1600)
            ground = np.column_stack((cx + r * np.cos(theta), cy + r * np.sin(theta)))
        else:
            ground = np.column_stack((np.linspace(0, 3.5, 1600), np.full(1600, y)))
        points = np.rint((ground - [x0, y0]) * ppm).astype(np.int32)
        cv2.polylines(texture, [points], False, (235, 235, 235), 14, cv2.LINE_AA)
    return RoadScene(texture, ppm, (x0, y0), (), ()).render(CAM, POSE), (cx, cy, radius, width)


class PhysicalM02Tests(unittest.TestCase):
    def test_m01_through_m02_validates_actual_pair_geometry(self):
        raw, _ = physical_scene()

        class Capture(RoadCapture):
            def image(self):
                return raw.copy()

        with CameraManager(CameraConfig(source=0), Capture) as camera:
            source = camera.wait_for_frame(timeout=1)
            before = source.image.copy()
            frame = BEVProcessor(PROFILE).process(source)
            result = LaneDetector(LaneConfig(min_lane_width_px=115, max_lane_width_px=135)).process(frame)
            self.assertTrue(result.valid, result.reason_codes)
            self.assertEqual((result.source_session_id, result.source_frame_id), (source.session_id, source.frame_id))
            self.assertEqual(result.acquired_monotonic_ns, source.acquired_monotonic_ns)
            self.assertFalse(result.metric_validated)
            pair = result.selected_pair
            boundaries = {b.candidate_id: b for b in result.boundaries}
            for identifier, truth in ((pair.left_id, .25), (pair.right_id, -.25)):
                xy = frame.geometry.bev_to_ground(boundaries[identifier].polynomial.samples())
                self.assertLess(np.max(np.abs(xy[:, 1] - truth)), .008)
            centre = frame.geometry.bev_to_ground(pair.centreline.samples())
            self.assertLess(np.max(np.abs(centre[:, 1])), .008)
            self.assertGreater(pair.confidence, .6)
            self.assertIsNone(pair.width_mean_m)
            np.testing.assert_array_equal(source.image, before)
            self.assertFalse(frame.image.flags.writeable)

    def test_actual_warp_of_constant_normal_width_arc_and_trust(self):
        raw, circle = physical_scene(curved=True)
        profile = fixture_profile()  # SIMULATED attestation tests software gates only
        frame = BEVProcessor(profile, ProcessorConfig(max_frame_age_s=None), report_for(profile)).process(packet(image=raw))
        result = LaneDetector(LaneConfig(lane_width_m=(.48, .52))).process(frame)
        self.assertTrue(result.valid, result.reason_codes)
        self.assertTrue(result.metric_validated)
        pair = result.selected_pair
        self.assertAlmostEqual(pair.width_mean_m, .5, delta=.01)
        cx, cy, radius, width = circle
        by_id = {b.candidate_id: b for b in result.boundaries}
        for identifier, r in ((pair.left_id, radius - width / 2), (pair.right_id, radius + width / 2)):
            xy = frame.geometry.bev_to_ground(by_id[identifier].polynomial.samples())
            self.assertLess(np.max(np.abs(np.linalg.norm(xy - [cx, cy], axis=1) - r)), .01)
        xy = frame.geometry.bev_to_ground(pair.centreline.samples())
        self.assertLess(np.max(np.abs(np.linalg.norm(xy - [cx, cy], axis=1) - radius)), .01)
        missing = replace(frame, valid_mask=np.zeros_like(frame.valid_mask), support_mask=np.zeros_like(frame.support_mask))
        self.assertFalse(LaneDetector().process(missing).valid)

    def test_m02_straight_to_arc_reference_cannot_hide_truncated_support(self):
        frame = BEVProcessor(PROFILE).process(packet(image=CURVE_IMG))
        result = LaneDetector().process(frame)
        self.assertTrue(result.valid, result.reason_codes)
        self.assertTrue(result.limited_support)
        self.assertIn("far_field_unsupported", result.reason_codes)
        pair = result.selected_pair
        self.assertGreater(pair.centreline.v_min, 270)
        self.assertGreater(pair.observed_span_px, 150)
        xy = frame.geometry.bev_to_ground(pair.centreline.samples())
        truth = np.where(xy[:, 0] <= 1.2, 0, 1 - np.sqrt(1 - (xy[:, 0] - 1.2) ** 2))
        self.assertLess(np.max(np.abs(xy[:, 1] - truth)), .008)
        self.assertTrue(np.isnan(pair.centreline.evaluate([pair.centreline.v_min - 1, pair.centreline.v_max + 1])).all())
        self.assertTrue(pair.unsupported_far_intervals)
        self.assertTrue(any(b.excluded_observation_intervals for b in result.boundaries))
