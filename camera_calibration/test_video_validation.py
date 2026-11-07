"""Portable video-validation tests using a temporary encoded source."""
import tempfile
import json
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from camera_calibration.video_validation import inspect_video, pixel_demo_matrix, process_video


class VideoValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = self.root / "input.mp4"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 48))
        if not writer.isOpened():
            self.skipTest("mp4v encoder unavailable")
        for n in range(12):
            frame = np.zeros((48, 64, 3), np.uint8)
            cv2.line(frame, (n, 0), (n, 47), (255, 255, 255), 2)
            writer.write(frame)
        writer.release()
        self.quad = np.array([[3, 3], [60, 3], [60, 44], [3, 44]], np.float64)

    def test_properties_full_eof_and_saved_outputs(self):
        info = inspect_video(self.video)
        self.assertEqual((info["width"], info["height"], info["frames_reported"]), (64, 48, 12))
        result = process_video(self.video, self.root / "out", quad=self.quad, output_size=(64, 48))
        self.assertEqual(result["frames_processed"], 12)
        self.assertEqual(result["m01_final_state"], "eof")
        self.assertEqual((result["first_frame_id"], result["last_frame_id"]), (1, 12))
        self.assertEqual(result["source_frames_skipped"], 0)
        self.assertEqual(len(result["source_sessions"]), 1)
        self.assertTrue(all(v == 12 for v in result["output_verified_frames"].values()))
        self.assertEqual(result["calibration_status"], "PIXEL DEMO - NO METRIC SCALE OR UNDISTORTION")
        records = [json.loads(line) for line in (self.root / "out" / "frame_metadata.jsonl").read_text().splitlines()]
        self.assertEqual([record["frame_id"] for record in records], list(range(1, 13)))
        self.assertEqual(len({record["session_id"] for record in records}), 1)
        self.assertTrue(all(record["acquired_unix_ns"] > 0 and record["acquired_monotonic_ns"] > 0 for record in records))
        source = cv2.VideoCapture(str(self.video)); self.assertTrue(source.read()[0]); source.release()

    def test_repeat_and_limit(self):
        for i in range(2):
            result = process_video(self.video, self.root / f"out{i}", quad=self.quad, output_size=(64, 48), limit=5)
            self.assertEqual(result["frames_processed"], 5)
            self.assertEqual(result["last_frame_id"], 5)

    def test_output_content_matches_the_source_frame_by_frame(self):
        # counts alone would pass shuffled or wrong frames: compare every BEV frame with an independent warp of the same source frame
        process_video(self.video, self.root / "out", quad=self.quad, output_size=(64, 48))
        H = pixel_demo_matrix(self.quad, 64, 48, (64, 48))
        src, bev = cv2.VideoCapture(str(self.video)), cv2.VideoCapture(str(self.root / "out" / "bev_only.mp4"))
        for i in range(12):
            (ok1, s), (ok2, b) = src.read(), bev.read()
            self.assertTrue(ok1 and ok2, f"frame {i}")
            expected = cv2.warpPerspective(s, H, (64, 48)).astype(np.float32)
            self.assertLess(np.abs(b.astype(np.float32) - expected).mean(), 12.0, f"frame {i}")
        src.release(); bev.release()

    def test_profile_mode_uses_the_m02_runtime(self):
        from camera_calibration.mapper import BEVMapper
        from camera_calibration.test_bev import PROFILE
        small = PROFILE.scaled_to(64, 48)               # SYNTHETIC profile at the clip's size: exercises the plumbing, not geometry
        path = small.save(self.root / "profile.json")
        result = process_video(self.video, self.root / "m02", profile_path=path)
        self.assertEqual(result["frames_processed"], 12)
        self.assertIn("M02 BEVProcessor", result["transform_engine"])
        self.assertEqual(result["calibration_state"], "METRIC_UNVERIFIED")
        mapper = BEVMapper(small)
        src, bev = cv2.VideoCapture(str(self.video)), cv2.VideoCapture(str(self.root / "m02" / "bev_only.mp4"))
        for i in range(12):
            (_, s), (ok, b) = src.read(), bev.read()
            self.assertTrue(ok)
            self.assertEqual(b.shape[:2], small.bev.shape)
            self.assertLess(np.abs(b.astype(np.float32) - mapper.warp(s).astype(np.float32)).mean(), 12.0, f"frame {i}")
        src.release(); bev.release()
        self.assertNotIn("M02 runtime NOT used", json.dumps(result))

    def test_partial_export_is_declared_in_the_metrics(self):
        full = process_video(self.video, self.root / "full", quad=self.quad, output_size=(64, 48))
        part = process_video(self.video, self.root / "part", quad=self.quad, output_size=(64, 48), limit=5)
        self.assertEqual((full["partial_export"], full["frame_limit"]), (False, None))
        self.assertEqual((part["partial_export"], part["frame_limit"]), (True, 5))
        self.assertLess(part["frames_processed"], part["input"]["frames_reported"])

    def test_metric_export_names_the_profile_and_flags_synthetic_geometry(self):
        from unittest.mock import patch
        import camera_calibration.video_validation as vv
        from camera_calibration.test_bev import PROFILE
        small = PROFILE.scaled_to(64, 48)                                    # origin text: SYNTHETIC
        lines = vv._provenance_lines(small)
        self.assertIn("SYNTHETIC GEOMETRY - NOT THIS CAMERA", lines)
        self.assertTrue(any("origin: SYNTHETIC" in line for line in lines))
        self.assertTrue(any(small.camera_id in line for line in lines))
        path = small.save(self.root / "profile.json")
        result = process_video(self.video, self.root / "with", profile_path=path)
        self.assertEqual(result["profile_origin_description"], "SYNTHETIC")
        self.assertEqual(result["profile_camera_id"], small.camera_id)
        with patch.object(vv, "_provenance_lines", return_value=()):
            process_video(self.video, self.root / "without", profile_path=path)
        shown = cv2.imread(str(self.root / "with" / "sample_frames" / "frame_0001.jpg"))
        bare = cv2.imread(str(self.root / "without" / "sample_frames" / "frame_0001.jpg"))
        w = small.bev.width_px
        caption_rows = np.abs(shown[40:100, w:].astype(np.int16) - bare[40:100, w:].astype(np.int16)).mean()
        self.assertGreater(caption_rows, 1.0, "the provenance caption is not drawn on the exported frame")

    def test_pixel_mode_says_m02_runtime_was_not_used(self):
        result = process_video(self.video, self.root / "px", quad=self.quad, output_size=(64, 48), limit=3)
        self.assertIn("M02 runtime NOT used", result["transform_engine"])
        self.assertEqual(result["calibration_state"], "PIXEL_ONLY")
        self.assertNotIn("m02_mean_ms", result)

    def test_pixel_overlay_does_not_mutate_m01_images(self):
        from unittest.mock import patch
        import camera_calibration.video_validation as vv
        original = vv.CameraManager.wait_for_frame
        held = []
        def capture(self_, *args, **kwargs):
            packet = original(self_, *args, **kwargs)
            held.append((packet, packet.image.copy()))
            return packet
        with patch.object(vv.CameraManager, "wait_for_frame", capture):
            process_video(self.video, self.root / "safe", quad=self.quad, output_size=(64, 48), limit=4)
        self.assertEqual(len(held), 4)
        self.assertTrue(all(np.array_equal(packet.image, before) for packet, before in held))

    def test_lost_first_frames_fail_instead_of_producing_a_silently_short_export(self):
        from unittest.mock import patch
        import camera_calibration.video_validation as vv
        real_enter = vv.CameraManager.__enter__
        def late(self_):
            # Deterministic, load-independent: return only once M01 has really published 3 frames, so 1-2 are gone.
            r = real_enter(self_)
            deadline = time.monotonic() + 10
            while self_.status()["frames"] < 3 and time.monotonic() < deadline:
                time.sleep(0.005)
            return r
        with patch.object(vv.CameraManager, "__enter__", late):
            with self.assertRaisesRegex(RuntimeError, "before this consumer read them.*persisted for 3 attempts"):
                process_video(self.video, self.root / "late", quad=self.quad, output_size=(64, 48))
        self.assertTrue((self.root / "late" / "RUN_INCOMPLETE.txt").is_file())

    def test_a_transient_start_up_race_is_retried_and_reported(self):
        from unittest.mock import patch
        import camera_calibration.video_validation as vv
        real_enter, calls = vv.CameraManager.__enter__, {"n": 0}
        def late_once(self_):
            r = real_enter(self_)
            calls["n"] += 1
            if calls["n"] == 1:                                       # only the first attempt starts late
                deadline = time.monotonic() + 10
                while self_.status()["frames"] < 3 and time.monotonic() < deadline:
                    time.sleep(0.005)
            return r
        with patch.object(vv.CameraManager, "__enter__", late_once):
            result = process_video(self.video, self.root / "retry", quad=self.quad, output_size=(64, 48))
        self.assertEqual(result["start_attempts"], 2)
        self.assertEqual((result["first_frame_id"], result["last_frame_id"], result["frames_processed"]), (1, 12, 12))
        self.assertTrue(all(v == 12 for v in result["output_verified_frames"].values()))
        self.assertFalse((self.root / "retry" / "RUN_INCOMPLETE.txt").exists())
        self.assertEqual(json.loads((self.root / "retry" / "processing_metrics.json").read_text())["start_attempts"], 2)

    def test_a_gap_after_the_first_frame_is_never_retried(self):
        from unittest.mock import patch
        import camera_calibration.video_validation as vv
        real_wait, state = vv.CameraManager.wait_for_frame, {"n": 0}
        def skipping_wait(self_, after, timeout=None, *, session_id=None):
            state["n"] += 1
            p = real_wait(self_, after, timeout=timeout, session_id=session_id)
            if state["n"] == 4:                                       # pretend frame 4 was missed: ask for anything after 4
                p = real_wait(self_, p.frame_id, timeout=timeout, session_id=session_id)
            return p
        with patch.object(vv.CameraManager, "wait_for_frame", skipping_wait):
            with self.assertRaisesRegex(RuntimeError, "published source frames 4..4"):
                process_video(self.video, self.root / "gap", quad=self.quad, output_size=(64, 48))
        self.assertEqual(state["n"], 4)                               # the 4th read found the gap; no retry once frames were produced

    def test_failed_run_is_marked_and_stale_metrics_cannot_vouch_for_it(self):
        from unittest.mock import patch
        import camera_calibration.video_validation as vv
        out = self.root / "reuse"
        process_video(self.video, out, quad=self.quad, output_size=(64, 48))
        self.assertTrue((out / "processing_metrics.json").is_file())
        self.assertFalse((out / "RUN_INCOMPLETE.txt").exists())
        real_warp, calls = vv.cv2.warpPerspective, {"n": 0}
        def failing(*a, **k):
            calls["n"] += 1
            if calls["n"] > 5: raise RuntimeError("injected failure")
            return real_warp(*a, **k)
        with patch.object(vv.cv2, "warpPerspective", failing), self.assertRaisesRegex(RuntimeError, "injected failure"):
            process_video(self.video, out, quad=self.quad, output_size=(64, 48))
        self.assertFalse((out / "processing_metrics.json").exists())            # the old run's metrics are gone
        self.assertIn("injected failure", (out / "RUN_INCOMPLETE.txt").read_text(encoding="utf-8"))
        process_video(self.video, out, quad=self.quad, output_size=(64, 48))    # a later successful run clears the marker
        self.assertFalse((out / "RUN_INCOMPLETE.txt").exists())
        self.assertEqual(json.loads((out / "processing_metrics.json").read_text())["frames_processed"], 12)

    def test_corrupt_or_wrong_size_export_is_rejected(self):
        from camera_calibration.video_validation import _verify_video
        bad = self.root / "corrupt.mp4"
        bad.write_bytes(b"not an mp4")
        with self.assertRaises(OSError):
            _verify_video(bad)
        with self.assertRaisesRegex(OSError, "wrong frame size"):
            _verify_video(self.video, (65, 48))

    def test_incomplete_export_is_marked(self):
        from unittest.mock import patch
        import camera_calibration.video_validation as vv
        info = inspect_video(self.video)
        info["frames_reported"] += 1
        out = self.root / "short"
        with patch.object(vv, "inspect_video", return_value=info), self.assertRaisesRegex(RuntimeError, "incomplete export"):
            process_video(self.video, out, quad=self.quad, output_size=(64, 48))
        self.assertFalse((out / "processing_metrics.json").exists())
        self.assertTrue((out / "RUN_INCOMPLETE.txt").exists())

    def test_samples_are_spread_over_the_actual_length(self):
        from camera_calibration.video_validation import _sample_positions
        self.assertEqual(_sample_positions(590), {0, 147, 294, 442, 589})
        self.assertEqual(_sample_positions(12), {0, 3, 6, 8, 11})
        self.assertEqual(_sample_positions(1), {0})
        process_video(self.video, self.root / "s", quad=self.quad, output_size=(64, 48))
        names = sorted(p.name for p in (self.root / "s" / "sample_frames").glob("*.jpg"))
        self.assertEqual(names, ["frame_0001.jpg", "frame_0004.jpg", "frame_0007.jpg", "frame_0009.jpg", "frame_0012.jpg"])

    def test_errors(self):
        with self.assertRaises(FileNotFoundError): inspect_video(self.root / "missing.mp4")
        with self.assertRaises(ValueError): process_video(self.video, self.root / "bad")
        with self.assertRaises(ValueError): process_video(self.video, self.root / "bad", quad=self.quad, output_size=(0, 48))
        with self.assertRaises(ValueError): process_video(self.video, self.root / "bad", profile_path=self.root / "missing.json")
        with self.assertRaises(ValueError): pixel_demo_matrix(np.zeros((4, 2)), 64, 48, (64, 48))


if __name__ == "__main__":
    unittest.main()
