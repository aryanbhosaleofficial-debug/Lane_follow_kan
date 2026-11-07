"""Integration tests: M02 runtime pipeline on top of the real M01 CameraManager (SYNTHETIC imagery, fake captures)."""
import gc
import threading
import time
import tracemalloc
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from camera_calibration.errors import (CalibrationError, FrameUnavailableError, InvalidCalibrationInputError, InvalidFrameError,
                                       ResolutionMismatchError, SourceMismatchError, StaleFrameError)
from camera_calibration.pipeline import BEVFrame, BEVProcessor, BEVWorker, ManagerFrameSource, ProcessorConfig
from camera_calibration.test_bev import BEV, PROFILE, STRAIGHT_IMG, build_profile
from camera_input import CameraConfig, CameraError, CameraManager, FramePacket


class RoadCapture:
    """Fake OpenCV capture serving the synthetic road image (a fresh array each read, like OpenCV)."""
    size = (640, 480)

    def __init__(self, source):
        self.source, self.n, self.closed = source, 0, False

    def isOpened(self): return True
    def set(self, *a): return True
    def get(self, prop): return 30
    def release(self): self.closed = True

    def read(self):
        time.sleep(0.004)
        self.n += 1
        return True, self.image()

    def image(self):
        return STRAIGHT_IMG.copy()


class SmallCapture(RoadCapture):
    def image(self): return np.zeros((240, 320, 3), np.uint8)


class WideCapture(RoadCapture):
    def image(self): return np.zeros((360, 640, 3), np.uint8)


def packet(frame_id=1, session="sess-A", source="0", image=None, age_s=0.0, **kw):
    img = STRAIGHT_IMG.copy() if image is None else image
    img.flags.writeable = False                               # M01 publishes read-only arrays
    now = time.monotonic_ns()
    h, w = img.shape[:2]
    base = dict(image=img, frame_id=frame_id, source_id=source, acquired_unix_ns=time.time_ns(),
                acquired_monotonic_ns=now - int(age_s * 1e9), width=w, height=h, session_id=session)
    base.update(kw)
    return FramePacket(**base)


class OldFrameSource:
    """Frame source whose frames were acquired 2 s ago, so staleness does not depend on the clock's resolution
    (Python <= 3.12 on Windows ticks its monotonic clock only every ~15.6 ms)."""
    def __init__(self):
        self.n = 0

    def next_frame(self, timeout):
        time.sleep(0.004)
        self.n += 1
        return packet(frame_id=self.n, age_s=2.0)


def started(factory=RoadCapture, source=0, **cfg):
    m = CameraManager(CameraConfig(source=source, **cfg), factory)
    m.start()
    return m


class ProcessorMetadataTests(unittest.TestCase):
    def setUp(self):
        self.proc = BEVProcessor(PROFILE)

    def test_m01_metadata_is_preserved_unchanged(self):
        p = packet(frame_id=42, session="abc123", source="0", presentation_ms=1234.5)
        f = self.proc.process(p)
        self.assertEqual((f.source_frame_id, f.source_session_id, f.source_id), (42, "abc123", "0"))
        self.assertEqual(f.key, ("abc123", 42))
        self.assertEqual((f.acquired_unix_ns, f.acquired_monotonic_ns), (p.acquired_unix_ns, p.acquired_monotonic_ns))
        self.assertEqual((f.source_width, f.source_height), (640, 480))
        self.assertEqual(f.source_presentation_ms, 1234.5)
        self.assertEqual(f.profile_fingerprint, PROFILE.fingerprint())
        self.assertEqual(f.pixel_format, "BGR")
        self.assertEqual(f.bev, BEV)
        self.assertEqual(f.meters_per_pixel, BEV.meters_per_pixel)

    def test_processing_timestamps_and_age(self):
        before = time.monotonic_ns()
        p = packet(age_s=0.05)
        f = self.proc.process(p)
        self.assertGreaterEqual(f.processed_monotonic_ns, before)
        self.assertGreater(f.processing_seconds, 0)
        self.assertGreaterEqual(f.age_seconds(), 0.05)                      # age counts from M01 acquisition, not from M02 processing
        self.assertLess(f.age_seconds(), 1.0)

    def test_output_arrays_are_read_only_and_shapes_match_the_grid(self):
        f = self.proc.process(packet())
        self.assertEqual(f.image.shape, (500, 350, 3))
        self.assertEqual(f.valid_mask.shape, (500, 350))
        for arr in (f.image, f.valid_mask, f.support_mask):
            self.assertFalse(arr.flags.writeable)
        with self.assertRaises(ValueError):
            f.image[0, 0, 0] = 1

    def test_geometry_travels_with_the_frame(self):
        f = self.proc.process(packet())
        g = f.geometry.ground_to_bev([[1.0, 0.0]])
        np.testing.assert_allclose(g, BEV.ground_to_bev([[1.0, 0.0]]))
        self.assertFalse(f.geometry.image_to_ground([[320.0, 400.0]]).size == 0)

    def test_input_image_is_never_modified(self):
        p = packet()
        before = p.image.copy()
        self.proc.process(p)
        np.testing.assert_array_equal(p.image, before)
        self.assertFalse(p.image.flags.writeable)

    def test_same_source_frame_is_not_reported_as_a_new_bev(self):
        p = packet(frame_id=7)
        a = self.proc.process(p)
        b = self.proc.process(p)
        c = self.proc.process(packet(frame_id=7))                           # equal identity, different packet object
        self.assertIs(a, b); self.assertIs(a, c)
        self.assertEqual(a.bev_sequence, 1)
        d = self.proc.process(packet(frame_id=8))
        self.assertEqual(d.bev_sequence, 2)
        self.assertEqual(self.proc.stats()["repeats"], 2)
        self.assertEqual(self.proc.stats()["processed"], 2)

    def test_repeated_identity_with_changed_source_is_rejected(self):
        self.proc.process(packet(frame_id=7, source="cam-a"))
        with self.assertRaisesRegex(InvalidFrameError, "conflicting source"):
            self.proc.process(packet(frame_id=7, source="cam-b"))

    def test_frame_id_going_backwards_within_a_session_is_rejected(self):
        self.proc.process(packet(frame_id=10, session="A"))
        with self.assertRaisesRegex(InvalidFrameError, "backwards"):
            self.proc.process(packet(frame_id=5, session="A"))
        later = self.proc.process(packet(frame_id=11, session="A"))              # the stream carries on after a refused frame
        self.assertEqual(later.source_frames_skipped, 0)
        self.assertEqual(self.proc.stats()["errors_by_type"].get("InvalidFrameError"), 1)

    def test_the_profile_is_not_hashed_for_every_frame(self):
        from unittest.mock import patch
        from camera_calibration.profile import CalibrationProfile
        expected = self.proc.process(packet(frame_id=1)).profile_fingerprint
        self.assertEqual(expected, self.proc.profile.fingerprint())
        with patch.object(CalibrationProfile, "fingerprint", side_effect=AssertionError("fingerprint() called per frame")):
            for fid in range(2, 8):
                self.assertEqual(self.proc.process(packet(frame_id=fid)).profile_fingerprint, expected)

    def test_same_frame_id_in_a_new_session_is_a_new_frame(self):
        a = self.proc.process(packet(frame_id=5, session="A"))
        b = self.proc.process(packet(frame_id=5, session="B"))
        self.assertIsNot(a, b)
        self.assertEqual(b.bev_sequence, 2)
        self.assertNotEqual(a.key, b.key)

    def test_frame_drops_are_reported(self):
        self.proc.process(packet(frame_id=1))
        f = self.proc.process(packet(frame_id=5))
        self.assertEqual(f.source_frames_skipped, 3)
        g = self.proc.process(packet(frame_id=6))
        self.assertEqual(g.source_frames_skipped, 0)
        h = self.proc.process(packet(frame_id=7, session="NEW"))
        self.assertIsNone(h.source_frames_skipped)                          # a new session has no meaningful gap
        self.assertEqual(self.proc.stats()["session_changes"], 1)

    def test_optional_undistorted_output(self):
        proc = BEVProcessor(PROFILE, ProcessorConfig(include_undistorted=True))
        f = proc.process(packet())
        self.assertEqual(f.undistorted_image.shape, (480, 640, 3))
        self.assertFalse(f.undistorted_image.flags.writeable)
        self.assertIsNone(self.proc.process(packet(frame_id=99)).undistorted_image)

    def test_duck_typed_packets_work_without_m01_classes(self):
        # M02 must run when M01 is absent: any object with the documented attributes is accepted
        obj = SimpleNamespace(image=STRAIGHT_IMG, frame_id=3, source_id="cam", acquired_unix_ns=time.time_ns(),
                              acquired_monotonic_ns=time.monotonic_ns(), width=640, height=480, session_id="s")
        f = self.proc.process(obj)
        self.assertEqual(f.source_id, "cam")


class ProcessorRejectionTests(unittest.TestCase):
    def setUp(self):
        self.proc = BEVProcessor(PROFILE)

    def test_invalid_packets_are_rejected(self):
        good = packet()
        cases = {
            "none": None,
            "float image": packet(image=np.zeros((480, 640, 3), np.float32)),
            "gray image": packet(image=np.zeros((480, 640), np.uint8), width=640, height=480),
            "4 channels": packet(image=np.zeros((480, 640, 4), np.uint8)),
            "empty image": SimpleNamespace(**{**good.__dict__, "image": np.zeros((0, 640, 3), np.uint8)}),
            "list image": SimpleNamespace(**{**good.__dict__, "image": [[1]]}),
            "metadata mismatch": packet(width=320),
            "non-numeric width": packet(width="640"),
            "boolean height": packet(height=True),
            "rgb format": packet(pixel_format="RGB"),
            "invalid flag": packet(valid=False),
            "negative frame id": packet(frame_id=-1),
            "bool frame id": packet(frame_id=True),
            "string frame id": packet(frame_id="3"),
            "missing field": SimpleNamespace(image=good.image, frame_id=1),
            "non-string session": SimpleNamespace(**{**good.__dict__, "session_id": 5}),
        }
        for label, p in cases.items():
            with self.subTest(label), self.assertRaises(InvalidFrameError):
                self.proc.process(p)
        self.assertEqual(self.proc.stats()["processed"], 0)

    def test_resolution_mismatch_is_an_error_not_a_scaled_guess(self):
        for img in (np.zeros((240, 320, 3), np.uint8), np.zeros((480, 639, 3), np.uint8), np.zeros((720, 1280, 3), np.uint8)):
            with self.subTest(shape=img.shape), self.assertRaises(ResolutionMismatchError):
                self.proc.process(packet(image=img))

    def test_opt_in_uniform_rescale(self):
        proc = BEVProcessor(PROFILE, ProcessorConfig(allow_uniform_rescale=True))
        f = proc.process(packet(image=np.zeros((240, 320, 3), np.uint8)))
        self.assertEqual((f.source_width, f.source_height), (320, 240))
        self.assertTrue(any("rescaled" in n for n in f.notes))
        self.assertEqual(f.image.shape, (500, 350, 3))                      # the BEV grid itself is unchanged
        with self.assertRaises(ResolutionMismatchError):
            proc.process(packet(frame_id=2, image=np.zeros((360, 640, 3), np.uint8)))     # aspect ratio differs: a crop, not a scale
        self.assertEqual(proc.process(packet(frame_id=3)).source_width, 640)              # full size still works

    def test_stale_frames_are_refused(self):
        with self.assertRaises(StaleFrameError):
            self.proc.process(packet(age_s=2.0))
        self.assertEqual(self.proc.stats()["stale"], 1)
        self.assertIsInstance(self.proc.process(packet(frame_id=2, age_s=0.1)), BEVFrame)
        lenient = BEVProcessor(PROFILE, ProcessorConfig(max_frame_age_s=None))
        self.assertIsInstance(lenient.process(packet(age_s=100.0)), BEVFrame)
        strict = BEVProcessor(PROFILE, ProcessorConfig(max_frame_age_s=0.01))
        with self.assertRaises(StaleFrameError):
            strict.process(packet(age_s=0.05))

    def test_source_binding_is_enforced_when_the_profile_is_bound(self):
        bound = BEVProcessor(PROFILE.with_source_ids("0"))
        bound.process(packet(source="0"))
        with self.assertRaises(SourceMismatchError):
            bound.process(packet(frame_id=2, source="1"))
        relaxed = BEVProcessor(PROFILE.with_source_ids("0"), ProcessorConfig(strict_source=False))
        relaxed.process(packet(source="1"))

    def test_session_change_alone_does_not_invalidate_the_profile(self):
        bound = BEVProcessor(PROFILE.with_source_ids("0"))
        a = bound.process(packet(frame_id=1, session="A", source="0"))
        b = bound.process(packet(frame_id=2, session="B", source="0"))
        self.assertNotEqual(a.source_session_id, b.source_session_id)

    def test_credentials_in_source_ids_never_reach_results_or_errors(self):
        url = "rtsp://admin:hunter2@10.0.0.5/stream"
        proc = BEVProcessor(PROFILE.with_source_ids("rtsp://***@10.0.0.5/stream"))
        f = proc.process(packet(source=url))                                # redacted form matches the bound id
        self.assertNotIn("hunter2", f.source_id)
        other = BEVProcessor(PROFILE.with_source_ids("0"))
        with self.assertRaises(SourceMismatchError) as ctx:
            other.process(packet(source=url))
        self.assertNotIn("hunter2", str(ctx.exception))

    def test_unmetric_profile_cannot_build_a_processor(self):
        with self.assertRaises(CalibrationError):
            BEVProcessor(build_profile(bev=None))

    def test_config_validation(self):
        for kw in ({"max_frame_age_s": 0}, {"max_frame_age_s": float("nan")}, {"max_frame_age_s": "1"}, {"strict_source": 1},
                   {"include_undistorted": None}, {"allow_uniform_rescale": "yes"}):
            with self.subTest(kw), self.assertRaises(InvalidCalibrationInputError):
                ProcessorConfig(**kw)

    def test_error_counters(self):
        for p in (packet(frame_id=1, age_s=5.0), packet(frame_id=2, image=np.zeros((10, 10, 3), np.uint8))):
            with self.assertRaises(CalibrationError):
                self.proc.process(p)
        s = self.proc.stats()
        self.assertEqual(s["errors"], 2)
        self.assertEqual(s["errors_by_type"], {"StaleFrameError": 1, "ResolutionMismatchError": 1})


class WithRealCameraManagerTests(unittest.TestCase):
    def test_bev_from_live_m01_frames_with_identity_preserved(self):
        cam = started()
        try:
            proc = BEVProcessor(PROFILE)
            src = ManagerFrameSource(cam)
            p = src.next_frame(2.0)
            f = proc.process(p)
            self.assertEqual((f.source_frame_id, f.source_session_id), (p.frame_id, p.session_id))
            self.assertEqual(f.source_id, "0")
            self.assertEqual(f.acquired_monotonic_ns, p.acquired_monotonic_ns)
            q = src.next_frame(2.0)
            self.assertGreater(q.frame_id, p.frame_id)                      # never the same frame twice
            self.assertGreater(proc.process(q).bev_sequence, f.bev_sequence)
        finally:
            cam.stop()

    def test_m02_does_not_consume_frames_from_other_m01_consumers(self):
        cam = started()
        try:
            proc = BEVProcessor(PROFILE)
            deadline = time.monotonic() + 2
            while cam.latest() is None and time.monotonic() < deadline: time.sleep(0.002)
            f = proc.process_latest(cam)
            # M02 used only the non-consuming interface: the legacy mailbox and latest() are untouched
            self.assertIsNotNone(cam.latest())
            legacy = cam.read()
            self.assertIsNotNone(legacy)
            self.assertGreaterEqual(legacy.frame_id, f.source_frame_id)
        finally:
            cam.stop()

    def test_several_independent_m02_consumers_and_a_third_party_all_get_frames(self):
        cam = started()
        results = {}
        def consume(name):
            proc, src, ids = BEVProcessor(PROFILE), ManagerFrameSource(cam), []
            for _ in range(8):
                ids.append(proc.process(src.next_frame(2.0)).source_frame_id)
            results[name] = ids
        def third_party():
            last, ids = 0, []
            for _ in range(8):
                p = cam.wait_for_frame(last, timeout=2.0); ids.append(p.frame_id); last = p.frame_id
            results["other"] = ids
        try:
            threads = [threading.Thread(target=consume, args=(n,)) for n in ("lane", "signs")] + [threading.Thread(target=third_party)]
            for t in threads: t.start()
            for t in threads: t.join(10)
            self.assertEqual(set(results), {"lane", "signs", "other"})
            for name, ids in results.items():
                self.assertEqual(len(ids), 8, name)
                self.assertTrue(all(b > a for a, b in zip(ids, ids[1:])), name)
        finally:
            cam.stop()

    def test_source_switch_changes_session_and_source_but_keeps_the_profile(self):
        cam = started(source=0)
        try:
            proc = BEVProcessor(PROFILE)                                     # unbound profile: any source of the right resolution
            src = ManagerFrameSource(cam)
            a = proc.process(src.next_frame(2.0))
            cam.switch_source(CameraConfig(source=1))
            b = proc.process(src.next_frame(2.0))
            self.assertEqual((a.source_id, b.source_id), ("0", "1"))
            self.assertNotEqual(a.source_session_id, b.source_session_id)
            self.assertGreater(b.source_frame_id, a.source_frame_id)         # M01 frame ids keep increasing across sessions
            self.assertEqual(proc.stats()["session_changes"], 1)
            self.assertEqual(proc.stats()["source_changes"], 1)
            self.assertIsNone(b.source_frames_skipped)
        finally:
            cam.stop()

    def test_switch_to_a_source_the_profile_is_not_bound_to_is_refused(self):
        cam = started(source=0)
        try:
            proc = BEVProcessor(PROFILE.with_source_ids("0"))
            src = ManagerFrameSource(cam)
            proc.process(src.next_frame(2.0))
            cam.switch_source(CameraConfig(source=1))
            with self.assertRaises(SourceMismatchError):
                proc.process(src.next_frame(2.0))
        finally:
            cam.stop()

    def test_switch_to_a_different_resolution_is_refused(self):
        def factory(source):
            return SmallCapture(source) if source == 1 else RoadCapture(source)
        cam = started(factory=factory)
        try:
            proc = BEVProcessor(PROFILE)
            src = ManagerFrameSource(cam)
            proc.process(src.next_frame(2.0))
            cam.switch_source(CameraConfig(source=1))
            with self.assertRaises(ResolutionMismatchError):
                proc.process(src.next_frame(2.0))
        finally:
            cam.stop()

    def test_missing_frames_and_stopped_camera(self):
        cam = CameraManager(CameraConfig(), RoadCapture)                       # never started
        proc = BEVProcessor(PROFILE)
        with self.assertRaises(FrameUnavailableError) as ctx:
            proc.process_latest(cam)
        self.assertEqual(ctx.exception.reason, "no_frame")
        with self.assertRaises(FrameUnavailableError) as ctx:
            ManagerFrameSource(cam).next_frame(0.1)
        self.assertEqual(ctx.exception.reason, "stopped")

    def test_camera_failure_is_reported_as_a_terminal_error(self):
        class Dying(RoadCapture):
            def read(self):
                self.n += 1
                return (True, self.image()) if self.n <= 3 else (False, None)
        cam = started(factory=Dying, retries=0)
        try:
            src = ManagerFrameSource(cam)
            with self.assertRaises(FrameUnavailableError) as ctx:
                for _ in range(50):
                    src.next_frame(1.0)
            self.assertEqual(ctx.exception.reason, "camera_error")
            self.assertIn("read failed", str(ctx.exception))
        finally:
            cam.stop()

    def test_video_end_is_propagated(self):
        class OneFrame(RoadCapture):
            def get(self, prop): return 1
            def read(self):
                if self.n: return False, None
                self.n += 1; return True, self.image()
        with patch("os.path.isfile", return_value=True):
            cam = CameraManager(CameraConfig(source="clip.mp4", kind="file", pace_video=False), OneFrame)
            cam.start()
            try:
                src = ManagerFrameSource(cam)
                self.assertIsNotNone(src.next_frame(2.0))
                with self.assertRaises(EOFError):
                    src.next_frame(2.0)
            finally:
                cam.stop()


class ManagerFrameSourceTests(unittest.TestCase):
    class ScriptedCamera:
        """Stand-in for M01 that reproduces its switch behaviour: 'Capture stopped' errors while switching."""
        def __init__(self, stopped_for_s):
            self.calls, self.state, self.t0 = 0, "stopping", time.monotonic()
            self.stopped_for_s = stopped_for_s

        def status(self):
            return {"state": self.state, "last_error": None}

        def wait_for_frame(self, after, timeout=None, session_id=None):
            self.calls += 1
            if time.monotonic() - self.t0 < self.stopped_for_s:
                raise CameraError("Capture stopped")
            return packet(frame_id=after + 1, session="new")

    def test_switch_window_is_retried_with_backoff_and_does_not_spin(self):
        cam = self.ScriptedCamera(0.15)
        p = ManagerFrameSource(cam, retry_interval_s=0.01).next_frame(2.0)
        self.assertEqual(p.frame_id, 1)
        self.assertLess(cam.calls, 40)                                          # ~15 retries, not a busy loop of thousands

    def test_stays_stopped_until_timeout(self):
        cam = self.ScriptedCamera(10.0)
        t0 = time.monotonic()
        with self.assertRaises(FrameUnavailableError) as ctx:
            ManagerFrameSource(cam).next_frame(0.1)
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertEqual(ctx.exception.reason, "stopped")

    def test_error_state_is_not_retried(self):
        cam = self.ScriptedCamera(10.0)
        cam.state = "error"
        t0 = time.monotonic()
        with self.assertRaises(FrameUnavailableError) as ctx:
            ManagerFrameSource(cam).next_frame(5.0)
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertEqual(ctx.exception.reason, "camera_error")
        self.assertEqual(cam.calls, 1)

    def test_timeouts_become_frame_unavailable(self):
        class Silent:
            def status(self): return {"state": "running"}
            def wait_for_frame(self, after, timeout=None, session_id=None):
                time.sleep(min(timeout, 0.02)); raise TimeoutError("nothing")
        with self.assertRaises(FrameUnavailableError) as ctx:
            ManagerFrameSource(Silent()).next_frame(0.1)
        self.assertEqual(ctx.exception.reason, "timeout")

    def test_argument_validation(self):
        with self.assertRaises(InvalidCalibrationInputError):
            ManagerFrameSource(object())
        cam = self.ScriptedCamera(0)
        for bad in (0, -1, float("nan"), "1"):
            with self.assertRaises(InvalidCalibrationInputError):
                ManagerFrameSource(cam).next_frame(bad)
        with self.assertRaises(InvalidCalibrationInputError):
            ManagerFrameSource(cam, retry_interval_s=0)


class WorkerTests(unittest.TestCase):
    def make(self, factory=RoadCapture, profile=PROFILE, config=None, **cam_kw):
        cam = started(factory, **cam_kw)
        worker = BEVWorker(BEVProcessor(profile, config), cam)
        self.addCleanup(cam.stop)
        self.addCleanup(worker.stop)
        return cam, worker

    def test_publishes_and_multiple_consumers_share_the_same_frame_objects(self):
        cam, worker = self.make()
        worker.start()
        first = worker.wait_for_bev(0, timeout=2)
        results = []
        def consume():
            results.append(worker.wait_for_bev(first.bev_sequence, timeout=2))
        threads = [threading.Thread(target=consume) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join(5)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r.bev_sequence > first.bev_sequence for r in results))
        by_sequence = {}
        for r in results:                                                       # consumers that woke on the same sequence got the SAME object
            self.assertIs(by_sequence.setdefault(r.bev_sequence, r), r)
        self.assertEqual(worker.status()["state"], "running")

    def test_slow_consumer_does_not_delay_the_worker_or_other_consumers(self):
        cam, worker = self.make()
        worker.start()
        slow_seen, fast_seen = [], []
        def slow():
            last = 0
            for _ in range(3):
                f = worker.wait_for_bev(last, timeout=2); slow_seen.append(f.bev_sequence); last = f.bev_sequence
                time.sleep(0.15)                                                # simulates a slow detector
        def fast():
            last = 0
            while len(fast_seen) < 15:
                f = worker.wait_for_bev(last, timeout=2); fast_seen.append(f.bev_sequence); last = f.bev_sequence
        ts = [threading.Thread(target=slow), threading.Thread(target=fast)]
        t0 = time.monotonic()
        for t in ts: t.start()
        for t in ts[1:]: t.join(5)
        fast_elapsed = time.monotonic() - t0
        ts[0].join(5)
        self.assertEqual(len(fast_seen), 15)
        self.assertLess(fast_elapsed, 1.2)                                       # the fast consumer was not held up by the slow one
        self.assertTrue(all(b > a for a, b in zip(fast_seen, fast_seen[1:])))
        self.assertGreater(slow_seen[-1] - slow_seen[0], 2)                      # the slow one skipped frames rather than queueing them

    def test_stop_wakes_waiters_and_is_deterministic_and_repeatable(self):
        cam, worker = self.make()
        worker.start()
        worker.wait_for_bev(0, timeout=2)
        outcome = []
        def waiter():
            try: worker.wait_for_bev(10 ** 9, timeout=5)
            except Exception as exc: outcome.append(exc)
        t = threading.Thread(target=waiter); t.start(); time.sleep(0.05)
        t0 = time.monotonic()
        worker.stop()
        t.join(2)
        self.assertFalse(t.is_alive())
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertIsInstance(outcome[0], FrameUnavailableError)
        self.assertEqual(outcome[0].reason, "stopped")
        self.assertEqual(worker.status()["state"], "stopped")
        self.assertIsNone(worker.latest())
        self.assertFalse(any(th.name == "bev-worker" and th.is_alive() for th in threading.enumerate()))
        worker.stop(); worker.stop()                                              # idempotent
        worker.start(); self.assertIsNotNone(worker.wait_for_bev(0, timeout=2))   # restartable

    def test_start_is_idempotent_and_context_manager_works(self):
        cam, worker = self.make()
        worker.start(); worker.start()
        self.assertEqual(sum(th.name == "bev-worker" for th in threading.enumerate()), 1)
        worker.stop()
        with BEVWorker(BEVProcessor(PROFILE), cam) as w:
            self.assertIsNotNone(w.wait_for_bev(0, timeout=2))
        self.assertEqual(w.status()["state"], "stopped")

    def test_source_switch_keeps_the_worker_running_with_new_session_frames(self):
        cam, worker = self.make()
        worker.start()
        a = worker.wait_for_bev(0, timeout=2)
        cam.switch_source(CameraConfig(source=1))
        deadline = time.monotonic() + 3
        b = a
        while b.source_session_id == a.source_session_id and time.monotonic() < deadline:
            b = worker.wait_for_bev(b.bev_sequence, timeout=2)
        self.assertNotEqual(a.source_session_id, b.source_session_id)
        self.assertEqual(b.source_id, "1")
        self.assertGreater(b.bev_sequence, a.bev_sequence)
        self.assertEqual(worker.status()["state"], "running")

    def test_old_frames_are_withdrawn_when_the_source_becomes_unusable_and_recovery_works(self):
        def factory(source):
            return SmallCapture(source) if source == 1 else RoadCapture(source)
        cam, worker = self.make(factory=factory)
        worker.start()
        good = worker.wait_for_bev(0, timeout=2)
        cam.switch_source(CameraConfig(source=1))                                # wrong resolution from now on
        # Until the switch takes effect the worker legitimately keeps publishing valid frames from the OLD source (more of
        # them the more the worker lags, e.g. under CPU load). Consume until the error arrives; nothing from the new source
        # may ever be delivered as a BEV.
        after, delivered, deadline = good.bev_sequence, [], time.monotonic() + 5
        with self.assertRaises(ResolutionMismatchError):
            while time.monotonic() < deadline:
                frame = worker.wait_for_bev(after, timeout=3)
                after = frame.bev_sequence
                delivered.append(frame)
        self.assertTrue(all(f.source_id == "0" and f.source_width == 640 for f in delivered))
        self.assertIsNone(worker.latest())                                       # the previous BEV must not pass for current
        self.assertEqual(worker.status()["state"], "running")
        self.assertIn("ResolutionMismatchError", worker.status()["last_error"])
        cam.switch_source(CameraConfig(source=0))                                # recovery
        deadline = time.monotonic() + 3
        recovered = None
        while recovered is None and time.monotonic() < deadline:
            try:
                recovered = worker.wait_for_bev(good.bev_sequence, timeout=1)
            except CalibrationError:
                pass
        self.assertIsNotNone(recovered)
        self.assertGreater(recovered.bev_sequence, good.bev_sequence)

    def test_camera_failure_ends_the_worker_in_error_state(self):
        class Dying(RoadCapture):
            def read(self):
                self.n += 1
                return (True, self.image()) if self.n <= 4 else (False, None)
        cam, worker = self.make(factory=Dying, retries=0)
        worker.start()
        with self.assertRaises(FrameUnavailableError) as ctx:
            for _ in range(20):
                worker.wait_for_bev(0 if worker.latest() is None else worker.latest().bev_sequence, timeout=2)
        self.assertEqual(ctx.exception.reason, "camera_error")
        self.assertEqual(worker.status()["state"], "error")
        self.assertIsNone(worker.latest())

    def test_end_of_video_keeps_the_final_bev(self):
        class Short(RoadCapture):
            def get(self, prop): return 3
            def read(self):
                if self.n >= 3: return False, None
                self.n += 1; return True, self.image()
        with patch("os.path.isfile", return_value=True):
            cam = CameraManager(CameraConfig(source="clip.mp4", kind="file", pace_video=False), Short)
            cam.start()
            self.addCleanup(cam.stop)
            worker = BEVWorker(BEVProcessor(PROFILE, ProcessorConfig(max_frame_age_s=None)), cam)
            self.addCleanup(worker.stop)
            worker.start()
            deadline = time.monotonic() + 3
            while worker.status()["state"] != "eof" and time.monotonic() < deadline: time.sleep(0.01)
            self.assertEqual(worker.status()["state"], "eof")
            final = worker.latest()
            self.assertIsNotNone(final)
            with self.assertRaises(EOFError):
                worker.wait_for_bev(final.bev_sequence, timeout=1)
            self.assertIs(worker.wait_for_bev(0, timeout=1), final)             # the final frame stays available to late consumers

    def test_unexpected_exception_surfaces_instead_of_hanging_consumers(self):
        cam, worker = self.make()
        with patch.object(worker.processor, "process", side_effect=RuntimeError("boom")):
            worker.start()
            with self.assertRaises(FrameUnavailableError):
                for _ in range(10):
                    worker.wait_for_bev(0, timeout=2)
        self.assertEqual(worker.status()["state"], "error")
        self.assertIn("boom", worker.status()["last_error"])

    def test_stale_frames_are_skipped_and_counted_not_published(self):
        worker = BEVWorker(BEVProcessor(PROFILE), OldFrameSource())          # every frame is 2 s old; the default limit is 0.5 s
        self.addCleanup(worker.stop)
        worker.start()
        time.sleep(0.15)
        self.assertIsNone(worker.latest())
        s = worker.status()
        self.assertGreater(s["stale_skipped"], 0)
        self.assertEqual(s["published"], 0)
        self.assertEqual(s["state"], "running")

    def test_argument_validation(self):
        cam, worker = self.make()
        for bad in (-1, True, 1.5, "1"):
            with self.assertRaises(InvalidCalibrationInputError):
                worker.wait_for_bev(bad, timeout=0.1)
        for bad in (-1.0, float("nan"), "1", True):
            with self.assertRaises(InvalidCalibrationInputError):
                worker.wait_for_bev(0, timeout=bad)
        with self.assertRaises(InvalidCalibrationInputError):
            BEVWorker(BEVProcessor(PROFILE), cam, wait_timeout_s=0)

    def test_wait_before_start_reports_stopped_and_timeouts_work(self):
        cam, worker = self.make()
        with self.assertRaises(FrameUnavailableError):
            worker.wait_for_bev(0, timeout=0.1)
        worker.start()
        with self.assertRaises(TimeoutError):
            worker.wait_for_bev(10 ** 9, timeout=0.05)


class ResourceBoundTests(unittest.TestCase):
    def test_memory_does_not_grow_over_many_frames(self):
        proc = BEVProcessor(PROFILE)
        for i in range(10):
            proc.process(packet(frame_id=i + 1))
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        for i in range(10, 210):
            proc.process(packet(frame_id=i + 1))
        gc.collect()
        grown = tracemalloc.get_traced_memory()[0] - base
        tracemalloc.stop()
        self.assertLess(grown, 3_000_000)                                       # holds ~one BEV (0.5 MB) + one source frame, never 200

    def test_log_noise_is_bounded_for_repeated_identical_errors(self):
        cam = started(factory=lambda s: SmallCapture(s))
        worker = BEVWorker(BEVProcessor(PROFILE), cam)
        self.addCleanup(cam.stop); self.addCleanup(worker.stop)
        with self.assertLogs("camera_calibration.pipeline", level="WARNING") as logs:
            worker.start()
            time.sleep(0.3)
            worker.stop()
        self.assertGreater(worker.status()["processing_errors"], 5)
        self.assertEqual(len(logs.output), 1)                                   # the same problem is logged once, not once per frame

    def test_stale_frame_warnings_are_rate_limited_despite_changing_messages(self):
        # the stale message embeds the frame number and age, so it differs on every frame
        worker = BEVWorker(BEVProcessor(PROFILE), OldFrameSource())
        self.addCleanup(worker.stop)
        with self.assertLogs("camera_calibration.pipeline", level="WARNING") as logs:
            worker.start()
            time.sleep(0.3)
            worker.stop()
        skipped = worker.status()["stale_skipped"]
        self.assertGreater(skipped, 10)
        self.assertLessEqual(len(logs.output), 1 + skipped // BEVWorker.LOG_EVERY)
        self.assertIn("StaleFrameError", logs.output[0])


if __name__ == "__main__":
    unittest.main()
