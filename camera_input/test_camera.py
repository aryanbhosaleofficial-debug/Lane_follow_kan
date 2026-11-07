import time
from pathlib import Path
import unittest
import numpy as np
from camera_input import CameraConfig, CameraError, CameraManager

class FakeCapture:
    def __init__(self, source): self.n = 0; self.closed = False
    def isOpened(self): return True
    def set(self, *args): return True
    def get(self, prop): return 30
    def read(self):
        time.sleep(.005); self.n += 1
        return True, np.zeros((10, 20, 3), dtype=np.uint8)
    def release(self): self.closed = True

class CameraTests(unittest.TestCase):
    def test_config(self):
        for kw in ({'width': 0, 'height': 2}, {'fps': -1}, {'retries': -1}, {'resize': (0, 2)}):
            with self.assertRaises(ValueError): CameraConfig(**kw)
    def test_capture_switch_and_stop(self):
        m = CameraManager(CameraConfig(source=0), FakeCapture)
        m.start(); time.sleep(.04)
        p = m.read()
        self.assertEqual((p.width, p.height, p.source_id, p.pixel_format), (20, 10, '0', 'BGR'))
        self.assertGreater(p.acquired_monotonic_ns, 0)
        self.assertGreater(m.status()['dropped'], 0)
        m.switch_source(CameraConfig(source=1, resize=(5, 6))); time.sleep(.04)
        p = m.read(); self.assertEqual((p.source_id, p.width, p.height), ('1', 5, 6))
        m.stop(); self.assertEqual(m.status()['state'], 'stopped')
        m.start(); m.stop()
    def test_missing_file(self):
        m = CameraManager(CameraConfig(source='no-such-file.mp4', kind='file'))
        with self.assertRaises(CameraError): m.start()
    def test_video(self):
        m = CameraManager(CameraConfig(source=str(Path(__file__).resolve().parent.parent / 'videos' / 'test.mp4'), kind='file', pace_video=False))
        m.start(); m.thread.join(20)
        self.assertFalse(m.thread.is_alive())
        self.assertEqual(m.status()['state'], 'eof')
        self.assertIsNotNone(m.read()); m.stop()

import threading
from unittest.mock import patch

class FailureTests(unittest.TestCase):
    def wait_state(self, manager, expected, timeout=2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and manager.status()['state'] not in expected:
            time.sleep(.002)
        self.assertIn(manager.status()['state'], expected)

    def test_invalid_camera_and_release(self):
        cap = FakeCapture(0)
        cap.isOpened = lambda: False
        m = CameraManager(CameraConfig(), lambda _: cap)
        with self.assertRaises(CameraError): m.start()
        self.assertTrue(cap.closed)
        self.assertEqual(m.status()['state'], 'error')

    def test_failure_retries_bounded_and_cleanup(self):
        captures = []
        class Broken(FakeCapture):
            def read(self): return False, None
        def factory(src):
            c = Broken(src); captures.append(c); return c
        m = CameraManager(CameraConfig(retries=2, retry_delay=.001, max_retry_delay=.002), factory)
        m.start(); self.wait_state(m, {'error'}); m.stop()
        self.assertEqual(len(captures), 3)
        self.assertTrue(all(c.closed for c in captures))
        self.assertEqual(m.status()['failures'], 3)

    def test_reconnect_recovers(self):
        captures = []
        class OnceBroken(FakeCapture):
            def read(self): return False, None
        def factory(src):
            c = OnceBroken(src) if not captures else FakeCapture(src)
            captures.append(c); return c
        m = CameraManager(CameraConfig(retries=2, retry_delay=.001, max_retry_delay=.002), factory)
        m.start()
        deadline = time.monotonic() + 1
        while m.status()['frames'] == 0 and time.monotonic() < deadline: time.sleep(.005)
        self.assertGreater(m.status()['frames'], 0)
        m.stop(); self.assertTrue(all(c.closed for c in captures))

    def test_failed_switch_drops_old_frame(self):
        def factory(src):
            c = FakeCapture(src)
            if src == 2: c.isOpened = lambda: False
            return c
        m = CameraManager(CameraConfig(), factory); m.start(); time.sleep(.02)
        with self.assertRaises(CameraError): m.switch_source(CameraConfig(source=2))
        self.assertIsNone(m.read()); self.assertEqual(m.status()['state'], 'error')
        m.stop()

    def test_worker_exception_releases(self):
        cap = FakeCapture(0)
        def fail(): raise RuntimeError('synthetic failure')
        cap.read = fail
        m = CameraManager(CameraConfig(), lambda _: cap)
        m.start(); self.wait_state(m, {'error'}); m.stop()
        self.assertTrue(cap.closed)

    def test_concurrent_stop_and_read(self):
        m = CameraManager(CameraConfig(), FakeCapture); m.start()
        errors = []
        def reader():
            try:
                for _ in range(1000): m.read(); m.status()
            except Exception as e: errors.append(e)
        t = threading.Thread(target=reader); t.start()
        m.stop(); t.join()
        self.assertFalse(errors); self.assertFalse(m.thread.is_alive())

    def test_invalid_frame_not_published(self):
        class Invalid(FakeCapture):
            def read(self): return True, np.zeros((10, 20), dtype=np.uint8)
        m = CameraManager(CameraConfig(retries=0), Invalid)
        m.start(); self.wait_state(m, {'error'})
        self.assertIsNone(m.read()); m.stop()

    def test_repeated_and_concurrent_lifecycle(self):
        m = CameraManager(CameraConfig(), FakeCapture)
        for _ in range(3):
            m.start(); m.stop()
            self.assertFalse(m.thread.is_alive())
        errors = []
        def cycle():
            try:
                for _ in range(5): m.start(); m.stop()
            except Exception as exc: errors.append(exc)
        threads = [threading.Thread(target=cycle) for _ in range(3)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(3)
        self.assertFalse(errors)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        m.stop()

    def test_blocked_read_timeout_and_later_cleanup(self):
        entered, unblock = threading.Event(), threading.Event()
        cap = FakeCapture(0)
        def blocked():
            entered.set(); unblock.wait(2)
            return True, np.zeros((2, 2, 3), dtype=np.uint8)
        cap.read = blocked
        m = CameraManager(CameraConfig(), lambda _: cap)
        m.start(); self.assertTrue(entered.wait(1))
        try:
            with self.assertRaises(CameraError): m.stop(.01)
            self.assertFalse(cap.closed)
        finally:
            unblock.set(); m.stop()
        self.assertTrue(cap.closed); self.assertIsNone(m.read())

    def test_retry_open_failure_and_cancellation(self):
        calls = []
        class NoFrames(FakeCapture):
            def read(self): return False, None
        def factory(src):
            calls.append(src)
            if len(calls) == 1: return NoFrames(src)
            cap = FakeCapture(src); cap.isOpened = lambda: False
            return cap
        m = CameraManager(CameraConfig(retries=10, retry_delay=.5, max_retry_delay=.5), factory)
        m.start()
        deadline = time.monotonic() + 1
        while m.status()['failures'] == 0 and time.monotonic() < deadline: time.sleep(.002)
        before = time.monotonic(); m.stop()
        self.assertLess(time.monotonic() - before, .3)
        self.assertEqual(len(calls), 1)

    def test_metadata_and_buffer(self):
        m = CameraManager(CameraConfig(), FakeCapture)
        wall_before, mono_before = time.time_ns(), time.monotonic_ns()
        m.start()
        try:
            deadline = time.monotonic() + 1
            while m.status()['frames'] < 3 and time.monotonic() < deadline: time.sleep(.005)
            packet = m.read()
            self.assertGreaterEqual(packet.acquired_unix_ns, wall_before)
            self.assertGreaterEqual(packet.acquired_monotonic_ns, mono_before)
            self.assertEqual(packet.image.shape, (packet.height, packet.width, 3))
            self.assertIsNone(packet.presentation_ms)
            self.assertGreaterEqual(packet.age_seconds(), 0)
            self.assertIsNone(m.read())
            self.assertGreater(m.status()['dropped'], 0)
            self.assertGreater(m.status()['rolling_fps'], 0)
        finally: m.stop()

    def test_thread_start_exception_releases_capture(self):
        cap = FakeCapture(0)
        m = CameraManager(CameraConfig(), lambda _: cap)
        with patch('threading.Thread.start', side_effect=RuntimeError('cannot start')):
            with self.assertRaises(RuntimeError): m.start()
        self.assertTrue(cap.closed)
        self.assertEqual(m.status()['state'], 'error')
        m.stop()

class ReviewRegressionTests(unittest.TestCase):
    """Regression tests for M01 review findings (M01-REV-xxx)."""

    def test_rev001_slow_open_does_not_block_readers(self):
        class SlowOpen(FakeCapture):
            def __init__(self, src):
                super().__init__(src); time.sleep(.5)
        m = CameraManager(CameraConfig(), SlowOpen)
        latency = []
        def consumer():
            time.sleep(.1)
            t = time.monotonic(); m.read(); m.status(); latency.append(time.monotonic() - t)
        t = threading.Thread(target=consumer); t.start()
        m.start(); t.join(); m.stop()
        self.assertLess(latency[0], .1)

    def test_rev002_stall_visible_in_status(self):
        stall, n = threading.Event(), [0]
        cap = FakeCapture(0)
        def read():
            n[0] += 1
            if n[0] > 10: stall.wait(3)
            time.sleep(.005); return True, np.zeros((2, 2, 3), dtype=np.uint8)
        cap.read = read
        m = CameraManager(CameraConfig(), lambda _: cap); m.start()
        try:
            deadline = time.monotonic() + 1
            while m.status()['frames'] < 10 and time.monotonic() < deadline: time.sleep(.005)
            healthy_fps = m.status()['rolling_fps']
            time.sleep(.6)
            st = m.status()
            self.assertGreater(st['seconds_since_last_frame'], .5)
            self.assertIsNotNone(st['last_frame_monotonic_ns'])
            self.assertLess(st['rolling_fps'], healthy_fps / 2)
        finally:
            stall.set(); m.stop()

    def test_rev003_restart_after_stop_timeout_is_not_silent(self):
        unblock = threading.Event(); cap = FakeCapture(0)
        def blocked():
            unblock.wait(2); return True, np.zeros((2, 2, 3), dtype=np.uint8)
        cap.read = blocked
        m = CameraManager(CameraConfig(), lambda _: cap); m.start(); time.sleep(.05)
        try:
            with self.assertRaises(CameraError): m.stop(.01)
            self.assertEqual(m.status()['state'], 'stopping')
            with self.assertRaises(CameraError): m.start()
        finally:
            unblock.set()
        m.thread.join(1)
        self.assertNotEqual(m.status()['state'], 'running')
        m.stop(); self.assertEqual(m.status()['state'], 'stopped')
        m.start(); self.assertEqual(m.status()['state'], 'running'); m.stop()

    def test_rev004_credentials_redacted(self):
        url = 'rtsp://admin:s3cret@10.0.0.5:554/stream'
        class Closed(FakeCapture):
            def isOpened(self): return False
        m = CameraManager(CameraConfig(source=url), Closed)
        with self.assertLogs('camera_input.core', 'ERROR') as logs:
            with self.assertRaises(CameraError) as ctx: m.start()
        self.assertNotIn('s3cret', str(ctx.exception))
        self.assertNotIn('s3cret', '\n'.join(logs.output))
        self.assertNotIn('s3cret', m.status()['source'])
        self.assertIn('10.0.0.5', m.status()['source'])
        m2 = CameraManager(CameraConfig(source=url), FakeCapture); m2.start()
        try:
            deadline = time.monotonic() + 1; p = None
            while p is None and time.monotonic() < deadline: p = m2.read(); time.sleep(.002)
            self.assertNotIn('s3cret', p.source_id)
        finally: m2.stop()

    def test_rev007_truncated_file_is_not_clean_eof(self):
        class Truncated(FakeCapture):
            def get(self, prop):
                import cv2
                return {cv2.CAP_PROP_FRAME_COUNT: 100, cv2.CAP_PROP_POS_FRAMES: self.n}.get(prop, 30)
            def read(self):
                if self.n >= 40: return False, None
                return super().read()
        with patch('os.path.isfile', return_value=True):
            m = CameraManager(CameraConfig(source='x.mp4', kind='file', pace_video=False), Truncated)
            m.start(); m.thread.join(2)
        self.assertEqual(m.status()['state'], 'error')
        self.assertIn('premature', m.status()['last_error'])
        m.stop()

    def test_rev007_zero_frame_file_is_error(self):
        class Empty(FakeCapture):
            def read(self): return False, None
        with patch('os.path.isfile', return_value=True):
            m = CameraManager(CameraConfig(source='x.mp4', kind='file'), Empty)
            m.start(); m.thread.join(2)
        self.assertEqual(m.status()['state'], 'error'); m.stop()

    def test_rev009_paced_rate_matches_nominal(self):
        class Nominal(FakeCapture):
            def get(self, prop): return 50.0
            def read(self): return True, np.zeros((2, 2, 3), dtype=np.uint8)
        with patch('os.path.isfile', return_value=True):
            m = CameraManager(CameraConfig(source='x.mp4', kind='file'), Nominal)
            m.start(); time.sleep(2.0); frames = m.status()['frames']; m.stop()
        self.assertAlmostEqual(frames / 2.0, 50.0, delta=50.0 * .04)

    def test_rev013_terminal_reason_survives_stop(self):
        class Broken(FakeCapture):
            def read(self): return False, None
        m = CameraManager(CameraConfig(retries=0), Broken); m.start(); m.thread.join(1)
        m.stop()
        self.assertIsNotNone(m.status()['last_error'])

class MultiConsumerTests(unittest.TestCase):
    def test_latest_is_independent_of_read(self):
        gate = threading.Event()
        cap = FakeCapture(0)
        def read():
            if cap.n:
                gate.wait(2)
                return False, None
            cap.n += 1
            return True, np.zeros((2, 2, 3), dtype=np.uint8)
        cap.read = read
        m = CameraManager(CameraConfig(), lambda _: cap)
        m.start()
        try:
            deadline = time.monotonic() + 1
            while m.latest() is None and time.monotonic() < deadline: time.sleep(.001)
            results = []
            threads = [threading.Thread(target=lambda: results.append(m.latest())) for _ in range(2)]
            for t in threads: t.start()
            for t in threads: t.join()
            self.assertIsNotNone(results[0]); self.assertIs(results[0], results[1])
            self.assertIs(m.read(), results[0]); self.assertIs(m.latest(), results[0])
        finally:
            gate.set(); m.stop()
        self.assertIsNone(m.latest())

    def test_session_changes_on_restart_and_switch(self):
        m = CameraManager(CameraConfig(), FakeCapture)
        try:
            ids = []
            for source in (0, 0, 1):
                if ids: m.switch_source(CameraConfig(source=source))
                else: m.start()
                deadline = time.monotonic() + 1
                while m.latest() is None and time.monotonic() < deadline: time.sleep(.001)
                packet = m.latest()
                self.assertIsNotNone(packet)
                self.assertEqual(packet.session_id, m.status()['session_id'])
                ids.append(packet.session_id)
            self.assertEqual(len(set(ids)), 3)
        finally: m.stop()

    def test_wait_timeout_and_stop_wakes(self):
        entered, gate = threading.Event(), threading.Event()
        cap = FakeCapture(0)
        def read():
            entered.set(); gate.wait(2); return False, None
        cap.read = read
        m = CameraManager(CameraConfig(), lambda _: cap); m.start(); self.assertTrue(entered.wait(1))
        with self.assertRaises(TimeoutError): m.wait_for_frame(timeout=.01)
        outcomes = []
        def wait():
            try: m.wait_for_frame(timeout=1)
            except Exception as exc: outcomes.append(exc)
        t = threading.Thread(target=wait); t.start()
        try:
            with self.assertRaises(CameraError): m.stop(.01)
            t.join(1); self.assertFalse(t.is_alive()); self.assertIsInstance(outcomes[0], CameraError)
        finally: gate.set(); m.stop()

    def test_wait_session_transition(self):
        m = CameraManager(CameraConfig(), FakeCapture); m.start()
        try:
            first = m.wait_for_frame(timeout=1)
            m.switch_source(CameraConfig(source=1))
            with self.assertRaisesRegex(CameraError, 'session changed'):
                m.wait_for_frame(first.frame_id, timeout=1, session_id=first.session_id)
            second = m.wait_for_frame(timeout=1)
            self.assertNotEqual(first.session_id, second.session_id)
        finally: m.stop()

    def test_invalid_configuration_types(self):
        for kwargs in ({'fps': True}, {'fps': '30'}, {'fps': float('nan')}, {'kind': []},
                       {'resize': [3, 4]}, {'resize': ({}, 4)}, {'width': True, 'height': 1},
                       {'retries': True}, {'retry_delay': '1'}, {'pace_video': 1}, {'source': []}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): CameraConfig(**kwargs)
        self.assertEqual(CameraConfig(source='0').source, 0)
        self.assertEqual(CameraConfig(source='0', kind='file').source, '0')

    def test_unsupported_format_does_not_reconnect(self):
        calls = []
        class Invalid(FakeCapture):
            def read(self): return True, np.zeros((2, 2), np.uint8)
        def factory(source):
            calls.append(source); return Invalid(source)
        m = CameraManager(CameraConfig(retries=5), factory); m.start(); m.thread.join(1)
        self.assertEqual(m.status()['state'], 'error')
        self.assertIn('unsupported frame format', m.status()['last_error'])
        self.assertEqual(len(calls), 1); m.stop()

    def test_backend_selection_and_backoff_validation(self):
        calls = []
        def factory(*args): calls.append(args); return FakeCapture(args[0])
        m = CameraManager(CameraConfig(source=3, backend=42, backoff_factor=1.5), factory)
        m.start(); m.stop(); self.assertEqual(calls, [(3, 42)])
        for kw in ({'backend': True}, {'backend': -1}, {'backoff_factor': 0.5}, {'backoff_factor': True}):
            with self.assertRaises(ValueError): CameraConfig(**kw)

    def test_waiters_share_frame_and_failure_wakes(self):
        publish, fail = threading.Event(), threading.Event()
        class Controlled(FakeCapture):
            def read(self):
                if self.n:
                    fail.wait(2); return False, None
                publish.wait(2); self.n += 1
                return True, np.zeros((2, 2, 3), np.uint8)
        m = CameraManager(CameraConfig(retries=0), Controlled); m.start()
        results = []
        def waiter(after=0):
            try: results.append(m.wait_for_frame(after, timeout=2))
            except Exception as exc: results.append(exc)
        threads = [threading.Thread(target=waiter) for _ in range(2)]
        try:
            for t in threads: t.start()
            publish.set()
            for t in threads: t.join(1)
            self.assertTrue(all(not t.is_alive() for t in threads))
            self.assertEqual(len(results), 2); self.assertIs(results[0], results[1])
            failure_waiter = threading.Thread(target=waiter, args=(results[0].frame_id,))
            failure_waiter.start(); fail.set(); failure_waiter.join(1)
            self.assertFalse(failure_waiter.is_alive()); self.assertIsInstance(results[-1], CameraError)
        finally:
            publish.set(); fail.set(); m.stop()

    def test_concurrent_switch_and_waiter(self):
        blocked, release = threading.Event(), threading.Event()
        class Old(FakeCapture):
            def read(self): blocked.set(); release.wait(2); return False, None
        def factory(src): return Old(src) if src == 0 else FakeCapture(src)
        m = CameraManager(CameraConfig(), factory); m.start(); self.assertTrue(blocked.wait(1))
        old_session = m.status()['session_id']; outcome = []
        def waiter():
            try: m.wait_for_frame(timeout=2, session_id=old_session)
            except Exception as exc: outcome.append(exc)
        t = threading.Thread(target=waiter); t.start()
        switch = threading.Thread(target=lambda: m.switch_source(CameraConfig(source=1)))
        switch.start()
        try:
            t.join(1); self.assertFalse(t.is_alive()); self.assertIsInstance(outcome[0], CameraError)
        finally:
            release.set(); switch.join(2)
        try:
            packet = m.wait_for_frame(timeout=1)
            self.assertEqual(packet.source_id, '1'); self.assertNotEqual(packet.session_id, old_session)
        finally: m.stop()

    def test_wait_at_eof(self):
        class OneFrame(FakeCapture):
            def get(self, prop): return 1
            def read(self):
                if self.n: return False, None
                self.n += 1; return True, np.zeros((2, 2, 3), np.uint8)
        with patch('os.path.isfile', return_value=True):
            m = CameraManager(CameraConfig(source='synthetic.mp4', kind='file', pace_video=False), OneFrame)
            m.start(); m.thread.join(1)
            packet = m.wait_for_frame(timeout=1)
            with self.assertRaises(EOFError): m.wait_for_frame(packet.frame_id, timeout=1, session_id=packet.session_id)
            m.stop()

    def test_backend_exception_redacts_credentials(self):
        url = 'rtsp://user:privatepass@example.test/stream'
        def factory(source): raise RuntimeError('backend failed for ' + source)
        m = CameraManager(CameraConfig(source=url), factory)
        with self.assertLogs('camera_input.core', level='ERROR') as logs:
            with self.assertRaises(CameraError) as exc: m.start()
        self.assertNotIn('privatepass', str(exc.exception))
        self.assertNotIn('privatepass', m.status()['last_error'])
        self.assertNotIn('privatepass', '\n'.join(logs.output))

    def test_extreme_fps_is_validation_error(self):
        with self.assertRaises(ValueError): CameraConfig(fps=10**10000)

    def test_release_failure_reports_terminal_error(self):
        class BrokenRelease(FakeCapture):
            def read(self): return False, None
            def release(self): raise RuntimeError('release broke')
        with patch('os.path.isfile', return_value=True):
            m = CameraManager(CameraConfig(source='synthetic.mp4', kind='file'), BrokenRelease)
            m.start(); m.thread.join(1)
        self.assertFalse(m.thread.is_alive())
        self.assertEqual(m.status()['state'], 'error')
        self.assertIn('release broke', m.status()['last_error'])
        self.assertIsNone(m.latest())
        with self.assertRaises(CameraError): m.wait_for_frame(timeout=0)
        m.stop()

    def test_release_failure_preserves_worker_error(self):
        class Broken(FakeCapture):
            def read(self): raise RuntimeError('read broke')
            def release(self): raise RuntimeError('release broke')
        m = CameraManager(CameraConfig(), Broken); m.start(); m.thread.join(1)
        self.assertIn('read broke', m.status()['last_error'])
        self.assertIn('release broke', m.status()['last_error'])
        m.stop()

    def test_cleanup_failure_during_thread_start(self):
        class BadRelease(FakeCapture):
            def release(self): raise RuntimeError('release failed')
        m = CameraManager(CameraConfig(), BadRelease)
        with patch('threading.Thread.start', side_effect=RuntimeError('start failed')):
            with self.assertRaisesRegex(RuntimeError, 'start failed'): m.start()
        self.assertEqual(m.status()['state'], 'error')
        self.assertIn('release failed', m.status()['last_error'])
        m.stop()

    def test_cleanup_failure_preserves_open_error(self):
        class BadOpen(FakeCapture):
            def isOpened(self): return False
            def release(self): raise RuntimeError('release failed')
        m = CameraManager(CameraConfig(), BadOpen)
        with self.assertRaisesRegex(CameraError, 'Cannot open source'): m.start()
        self.assertEqual(m.status()['state'], 'error')
        m.stop()

class SecondReviewRegressionTests(unittest.TestCase):
    """Regression tests for the second M01 review (M01-R2-xxx)."""

    def test_r2_001_shared_image_is_read_only(self):
        import cv2
        m = CameraManager(CameraConfig(), FakeCapture); m.start()
        try:
            a = m.wait_for_frame(timeout=1); b = m.latest()
            self.assertIs(a.image, b.image)
            with self.assertRaises(ValueError): a.image[:] = 255
            with self.assertRaises(cv2.error): cv2.line(a.image, (0, 0), (3, 3), (0, 0, 255), 1)
            self.assertEqual(int(b.image.max()), 0)
            copy = a.image.copy(); copy[:] = 1  # documented way to get a mutable image
        finally: m.stop()

    def test_r2_002_restart_while_failed_worker_is_exiting(self):
        releasing, opened = threading.Event(), []
        class SlowRelease(FakeCapture):
            def read(self):
                if len(opened) == 1: return False, None
                return super().read()
            def release(self): releasing.set(); time.sleep(.3); self.closed = True
        m = CameraManager(CameraConfig(retries=0), lambda s: (opened.append(1), SlowRelease(s))[1])
        m.start()
        try:
            self.assertTrue(releasing.wait(1))
            self.assertEqual(m.status()['state'], 'error')
            old_session = m.status()['session_id']
            m.start()  # must not be silently ignored
            self.assertEqual(len(opened), 2)
            self.assertEqual(m.status()['state'], 'running')
            self.assertNotEqual(m.status()['session_id'], old_session)
            self.assertGreater(m.wait_for_frame(timeout=1).frame_id, 0)
        finally: m.stop()

    def test_r2_003_no_frame_published_after_stop_request(self):
        import cv2
        m = CameraManager(CameraConfig(resize=(2, 2)), FakeCapture)
        real_resize = cv2.resize
        class SlowCv:
            def __getattr__(self, name): return getattr(cv2, name)
            def resize(self, image, size):
                m.stop_event.wait(2); time.sleep(.2); return real_resize(image, size)
        m.cv = SlowCv(); m.start()
        try:
            time.sleep(.05)
            with self.assertRaises(CameraError): m.stop(.02)
            m.thread.join(2); self.assertFalse(m.thread.is_alive())
            self.assertIsNone(m.latest()); self.assertIsNone(m.read())
            self.assertNotEqual(m.status()['state'], 'running')
        finally: m.stop()

    def test_r2_unplugged_camera_reopen_failures_are_bounded(self):
        # First capture streams then fails; every reopen fails (device gone). Must end in error with bounded opens.
        caps = []
        def factory(src):
            cap = FakeCapture(src)
            if not caps:  # the real camera: three good frames, then the cable is pulled
                def read_then_fail():
                    cap.n += 1
                    return (True, np.zeros((2, 2, 3), dtype=np.uint8)) if cap.n <= 3 else (False, None)
                cap.read = read_then_fail
            else:  # device no longer enumerates
                cap.isOpened = lambda: False
            caps.append(cap); return cap
        m = CameraManager(CameraConfig(retries=3, retry_delay=.001, max_retry_delay=.004), factory)
        m.start()
        try:
            m.thread.join(2); self.assertFalse(m.thread.is_alive())
            st = m.status()
            self.assertEqual(st['state'], 'error')
            self.assertIn('reconnection failed after 3 attempts', st['last_error'])
            self.assertEqual(len(caps), 1 + 3)  # initial open + exactly `retries` reopen attempts
            self.assertTrue(all(c.closed for c in caps))
            self.assertIsNone(m.latest())
            with self.assertRaises(CameraError): m.wait_for_frame(timeout=0)
        finally: m.stop()

    def test_r2_three_consumers_slow_and_stopped(self):
        m = CameraManager(CameraConfig(), FakeCapture); m.start(); seen = {}
        try:
            first = m.wait_for_frame(timeout=1)
            def fast(name):
                last, ids = first.frame_id, []
                for _ in range(10):
                    p = m.wait_for_frame(last, timeout=1, session_id=first.session_id)
                    ids.append(p.frame_id); last = p.frame_id
                seen[name] = ids
            def slow():
                p = m.wait_for_frame(first.frame_id, timeout=1); time.sleep(.1); seen['slow'] = [p.frame_id]
            threads = [threading.Thread(target=fast, args=(i,)) for i in range(3)] + [threading.Thread(target=slow)]
            before = m.status()['frames']
            for t in threads: t.start()
            for t in threads: t.join(3)
            self.assertTrue(all(not t.is_alive() for t in threads))
            for i in range(3):
                self.assertEqual(len(seen[i]), 10)
                self.assertTrue(all(b > a for a, b in zip(seen[i], seen[i][1:])))
            self.assertTrue(set(seen[0]) & set(seen[1]) & set(seen[2]))  # same frames observed by all
            self.assertGreater(m.status()['frames'] - before, 10)  # slow consumer did not stall acquisition
        finally: m.stop()

if __name__ == '__main__': unittest.main()
