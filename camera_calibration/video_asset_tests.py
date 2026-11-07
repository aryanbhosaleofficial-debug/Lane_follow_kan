"""Explicit real-project-video tests: python -m unittest camera_calibration.video_asset_tests -v.

Opt-in (not matched by the ``test_*.py`` discovery pattern) because they replay the recording at its nominal frame rate.
"""
import hashlib
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from camera_calibration.video_validation import inspect_video, process_video

VIDEO = Path(__file__).resolve().parent.parent / "videos" / "test.mp4"


def placeholder_profile(width, height):
    """PLACEHOLDER GEOMETRY for plumbing tests only: arbitrary focal length / pose, NOT a calibration of the project camera.
    No physical camera parameters for this recording exist; this profile only lets real frames flow through M02's runtime."""
    from camera_calibration import BEVConfig, RoadPlaneCalibrator, RoadPlaneConfig, new_profile
    from camera_calibration.synthetic import SyntheticCamera, SyntheticPose, project_ground_points
    cam = SyntheticCamera(width=width, height=height, fx=600.0, fy=600.0, cx=(width - 1) / 2, cy=(height - 1) / 2, dist=(0, 0, 0, 0, 0))
    pose = SyntheticPose(0.20, 25.0)
    ground = np.column_stack([m.ravel() for m in np.meshgrid(np.linspace(0.5, 1.5, 4), np.linspace(-0.3, 0.3, 3))])
    base = new_profile(cam.intrinsics(), "PLACEHOLDER-GEOMETRY-plumbing-test", note="arbitrary numbers; not a calibration")
    fit = RoadPlaneCalibrator(base.intrinsics, base.undistortion, RoadPlaneConfig(origin_description="placeholder")).estimate(
        project_ground_points(cam, pose, ground), ground)
    return base.with_road_plane(fit).with_bev(BEVConfig.from_extents(0.3, 2.0, -0.6, 0.6, 0.004))


@unittest.skipUnless(VIDEO.is_file(), "project video asset is absent")
class ProjectVideoTests(unittest.TestCase):
    def test_full_real_video(self):
        info = inspect_video(VIDEO)
        with tempfile.TemporaryDirectory(prefix="m02_asset_test_") as tmp:
            quad = np.array([[365, 20], [700, 20], [863, 495], [0, 495]], np.float64)
            result = process_video(VIDEO, Path(tmp), quad=quad)
        self.assertEqual(result["frames_processed"], info["frames_reported"])
        self.assertEqual((result["first_frame_id"], result["last_frame_id"]), (1, info["frames_reported"]))
        self.assertEqual(result["m01_final_state"], "eof")
        self.assertEqual(result["source_frames_skipped"], 0)
        self.assertTrue(all(v == result["frames_processed"] for v in result["output_verified_frames"].values()))

    def test_real_frames_through_the_m02_runtime_alongside_another_m01_consumer(self):
        """Real frames -> M01 -> M02 BEVWorker, with an unrelated M01 consumer: complete, ordered, bit-exact, unmutated."""
        from camera_input import CameraConfig, CameraManager
        from camera_calibration import BEVMapper, BEVProcessor, BEVWorker, ProcessorConfig
        n = 120
        info = inspect_video(VIDEO)
        profile = placeholder_profile(info["width"], info["height"])
        mapper = BEVMapper(profile)
        cap, src = cv2.VideoCapture(str(VIDEO)), []
        while len(src) < n:
            ok, f = cap.read()
            self.assertTrue(ok)
            src.append(f)
        cap.release()
        camera = CameraManager(CameraConfig(source=str(VIDEO), kind="file", pace_video=True))
        worker = BEVWorker(BEVProcessor(profile, ProcessorConfig(max_frame_age_s=None)), camera)
        other_ids, held, mutated = [], [], []

        def other_consumer():
            last, sess = 0, None
            while last < n:
                p = camera.wait_for_frame(last, timeout=5, session_id=sess)
                held.append((p, hashlib.sha1(p.image.tobytes()).hexdigest()))
                other_ids.append(p.frame_id)
                last, sess = p.frame_id, p.session_id

        camera.start()
        t = threading.Thread(target=other_consumer)
        t.start()
        worker.start()
        bevs, after = [], 0
        try:
            while len(bevs) < n:
                b = worker.wait_for_bev(after, timeout=5)
                bevs.append(b)
                after = b.bev_sequence
        finally:
            t.join(10)
            worker.stop()
            camera.stop()
        time.sleep(0.05)
        mutated = [p.frame_id for p, h in held if hashlib.sha1(p.image.tobytes()).hexdigest() != h]
        self.assertEqual([b.source_frame_id for b in bevs], list(range(1, n + 1)))
        self.assertEqual(sum(b.source_frames_skipped or 0 for b in bevs), 0)
        self.assertEqual(len({b.source_session_id for b in bevs}), 1)
        self.assertTrue(all(b.calibration_status.value == "METRIC_UNVERIFIED" for b in bevs))
        self.assertEqual(other_ids[:n], list(range(1, n + 1)))
        self.assertEqual(mutated, [])
        for b in bevs:
            self.assertTrue(np.array_equal(b.image, mapper.warp(src[b.source_frame_id - 1])), f"frame {b.source_frame_id}")


if __name__ == "__main__":
    unittest.main()
