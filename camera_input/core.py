"""Bounded, threaded OpenCV frame acquisition."""
from __future__ import annotations
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)

FPS_WINDOW_NS = 1_000_000_000
_URL_USERINFO = re.compile(r'(?<=://)[^/@]*@')

def redact_source(source: int | str) -> str:
    """Display form of a source with URL credentials removed."""
    return _URL_USERINFO.sub('***@', str(source))

class CameraError(RuntimeError):
    """A source could not be opened or operated."""

@dataclass(frozen=True)
class CameraConfig:
    source: int | str = 0
    kind: str = 'live'  # live or file
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    resize: tuple[int, int] | None = None
    retries: int = 3
    retry_delay: float = 0.25
    max_retry_delay: float = 2.0
    pace_video: bool = True
    slow_interval: float = 0.5
    backend: int | None = None
    backoff_factor: float = 2.0

    def __post_init__(self):
        def positive_number(value):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                return False
            try:
                return math.isfinite(value)
            except OverflowError:
                return False
        def dimension(value):
            return isinstance(value, int) and not isinstance(value, bool) and value > 0
        if not isinstance(self.kind, str) or self.kind not in ('live', 'file'):
            raise ValueError('kind must be live or file')
        if isinstance(self.source, bool) or not isinstance(self.source, (int, str)) or self.source == '':
            raise ValueError('source must be a nonempty string or nonnegative camera index')
        if isinstance(self.source, int) and self.source < 0:
            raise ValueError('camera index must be nonnegative')
        if self.kind == 'file' and not isinstance(self.source, str):
            raise ValueError('file source must be a path')
        if self.kind == 'live' and isinstance(self.source, str) and self.source.isascii() and self.source.isdecimal():
            object.__setattr__(self, 'source', int(self.source))
        if (self.width is None) != (self.height is None) or (self.width is not None and (not dimension(self.width) or not dimension(self.height))):
            raise ValueError('width and height must both be positive integers')
        if self.resize is not None and (not isinstance(self.resize, tuple) or len(self.resize) != 2 or not all(dimension(v) for v in self.resize)):
            raise ValueError('resize must be a tuple of two positive integers')
        if self.fps is not None and not positive_number(self.fps):
            raise ValueError('fps must be finite and positive')
        if not isinstance(self.retries, int) or isinstance(self.retries, bool) or self.retries < 0:
            raise ValueError('retries must be a nonnegative integer')
        if not positive_number(self.retry_delay) or not positive_number(self.max_retry_delay) or self.retry_delay > self.max_retry_delay or not positive_number(self.slow_interval):
            raise ValueError('invalid retry or monitoring settings')
        if self.backend is not None and (not isinstance(self.backend, int) or isinstance(self.backend, bool) or self.backend < 0):
            raise ValueError('backend must be a nonnegative OpenCV backend ID or None')
        if not positive_number(self.backoff_factor) or self.backoff_factor < 1:
            raise ValueError('backoff_factor must be finite and at least 1')
        if not isinstance(self.pace_video, bool):
            raise ValueError('pace_video must be a boolean')

@dataclass(frozen=True)
class FramePacket:
    image: Any
    frame_id: int
    source_id: str
    acquired_unix_ns: int
    acquired_monotonic_ns: int
    width: int
    height: int
    pixel_format: str = 'BGR'
    presentation_ms: float | None = None
    valid: bool = True
    session_id: str = ''
    def age_seconds(self) -> float:
        return max(0, (time.monotonic_ns() - self.acquired_monotonic_ns) / 1e9)

class CameraManager:
    def __init__(self, config: CameraConfig, capture_factory: Callable | None = None):
        import cv2
        self.cv = cv2
        self.config = config
        self.factory = capture_factory or cv2.VideoCapture
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.lifecycle_lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.capture = None
        self.packet = None
        self._latest = None
        self.state = 'stopped'
        self.sequence = 0
        self.session_id = None
        self.failures = self.dropped = self.slow_reads = 0
        self.frame_times = deque(maxlen=121)  # monotonic ns of recent frames, current session only
        self.last_error = None
        self.actual = {}

    def _open(self):
        c = self.config
        if c.kind == 'file' and not os.path.isfile(c.source):
            raise CameraError(f'Video file does not exist: {redact_source(c.source)}')
        try:
            cap = self.factory(c.source) if c.backend is None else self.factory(c.source, c.backend)
        except Exception as exc:
            raise CameraError(f'Cannot open source {redact_source(c.source)!r}: {redact_source(str(exc))}') from None
        try:
            if not cap.isOpened():
                raise CameraError(f'Cannot open source {redact_source(c.source)!r}; check availability, permissions, backend and codec (backend={c.backend})')
            if c.kind == 'live':
                for prop, val in ((self.cv.CAP_PROP_FRAME_WIDTH, c.width), (self.cv.CAP_PROP_FRAME_HEIGHT, c.height), (self.cv.CAP_PROP_FPS, c.fps), (self.cv.CAP_PROP_BUFFERSIZE, 1)):
                    if val is not None: cap.set(prop, val)
            actual = {'width': cap.get(self.cv.CAP_PROP_FRAME_WIDTH), 'height': cap.get(self.cv.CAP_PROP_FRAME_HEIGHT), 'fps': cap.get(self.cv.CAP_PROP_FPS)}
            with self.lock:
                self.actual = actual
            self.capture = cap
        except BaseException as exc:
            try:
                cap.release()
            except Exception as release_exc:
                log.error('Capture release after initialization failure failed: %s', redact_source(str(release_exc)))
            if isinstance(exc, Exception) and not isinstance(exc, CameraError):
                raise CameraError(f'Cannot initialize source {redact_source(c.source)!r}: {redact_source(str(exc))}') from None
            raise

    def start(self):
        with self.lifecycle_lock:
            self._start()

    def _start(self):
        # Caller holds lifecycle_lock. The data lock is NOT held while opening:
        # opening a device or stream can block for seconds and must not stall read()/status().
        if self.thread and self.thread.is_alive():
            if self.stop_event.is_set():
                raise CameraError('Previous capture thread is still stopping; call stop() again after the backend read returns')
            with self.lock:
                exiting = self.state not in ('running', 'starting')
            if not exiting:
                return  # already running: start() is idempotent
            # The worker has reached eof/error and is still releasing its capture; wait for it
            # instead of silently ignoring the restart request.
            self.thread.join(5)
            if self.thread.is_alive():
                raise CameraError('Previous capture thread is still releasing its source; retry start() later')
        self.stop_event.clear()
        with self.lock:
            self.packet = None
            self._latest = None
            self.session_id = uuid.uuid4().hex
            self.frame_times.clear()
            self.actual = {}
            self.state = 'starting'
            self.condition.notify_all()
        try: self._open()
        except Exception as exc:
            with self.lock:
                self.state = 'error'
                self.last_error = str(exc)
                self.condition.notify_all()
            log.exception('Failed to open camera source')
            raise
        with self.lock:
            self.state = 'running'
            self.last_error = None
        self.thread = threading.Thread(target=self._run, name='camera-acquisition', daemon=True)
        try:
            self.thread.start()
        except BaseException as exc:
            release_error = None
            try:
                self.capture.release()
            except Exception as release_exc:
                release_error = redact_source(str(release_exc))
                log.error('Capture release after thread startup failure failed: %s', release_error)
            self.capture = None
            self.thread = None
            with self.lock:
                self.state = 'error'
                self.last_error = f'Cannot start capture thread: {redact_source(str(exc))}'
                if release_error:
                    self.last_error += f'; capture release failed: {release_error}'
                self.condition.notify_all()
            raise

    def _fail(self, state: str, message: str):
        with self.lock:
            self.state = state
            if state == 'error':
                self.packet = None
                self._latest = None
            self.last_error = message
            self.condition.notify_all()

    def _run(self):
        retries = 0
        retry_delay = self.config.retry_delay
        session_frames = 0
        next_due = None
        try:
            while not self.stop_event.is_set():
                start = time.monotonic()
                ok, image = self.capture.read()
                mono, wall = time.monotonic_ns(), time.time_ns()
                if self.stop_event.is_set(): break
                if not ok or image is None or getattr(image, 'size', 0) == 0 or len(image.shape) != 3 or image.shape[2] != 3 or str(image.dtype) != 'uint8':
                    bad_format = ok and image is not None and getattr(image, 'size', 0) != 0
                    reason = (f'unsupported frame format shape={getattr(image, "shape", None)} dtype={getattr(image, "dtype", None)}; expected HxWx3 uint8 BGR'
                              if bad_format else 'read failed')
                    if bad_format and self.config.kind == 'live':
                        with self.lock: self.failures += 1
                        self._fail('error', reason)
                        log.error('Source %r: %s', redact_source(self.config.source), reason)
                        break
                    if self.config.kind == 'file':
                        if bad_format:
                            with self.lock: self.failures += 1
                            self._fail('error', reason)
                        elif session_frames == 0:
                            with self.lock: self.failures += 1
                            self._fail('error', 'file contains no readable frames')
                        else:
                            total = self.capture.get(self.cv.CAP_PROP_FRAME_COUNT)
                            # FRAME_COUNT can be a container estimate; allow a small tolerance.
                            if 0 < total < float('inf') and session_frames < total - max(2, 0.01 * total):
                                with self.lock: self.failures += 1
                                self._fail('error', f'premature end of file: decoded {session_frames} of {int(total)} frames (truncated or corrupt)')
                            else:
                                self._fail('eof', None)
                        break
                    with self.lock:
                        self.failures += 1
                        self.packet = None
                        self._latest = None
                        self.last_error = reason
                    if retries >= self.config.retries:
                        self._fail('error', f'{reason}; gave up after {self.config.retries} reconnection attempts')
                        log.error('Source %r failed after retries: %s', redact_source(self.config.source), reason)
                        break
                    self.capture.release(); self.capture = None
                    if self.stop_event.wait(retry_delay): break
                    retry_delay = min(retry_delay * self.config.backoff_factor, self.config.max_retry_delay)
                    retries += 1
                    try: self._open()
                    except Exception:
                        log.exception('Reconnection failed')
                        # Continue retrying without attempting a read on a missing capture.
                        while self.capture is None and retries < self.config.retries and not self.stop_event.is_set():
                            if self.stop_event.wait(retry_delay): break
                            retry_delay = min(retry_delay * self.config.backoff_factor, self.config.max_retry_delay)
                            retries += 1
                            try: self._open()
                            except Exception: log.exception('Reconnection failed')
                        if self.capture is None:
                            self._fail('error', f'{reason}; reconnection failed after {retries} attempts')
                            break
                    continue
                retries = 0
                retry_delay = self.config.retry_delay
                session_frames += 1
                if self.config.resize: image = self.cv.resize(image, self.config.resize)
                pts = self.capture.get(self.cv.CAP_PROP_POS_MSEC) if self.config.kind == 'file' else None
                if pts is not None and (not 0 <= pts < float('inf')): pts = None
                h, w = image.shape[:2]
                # Packets are shared by every consumer: make accidental in-place writes fail loudly.
                # This guards against mistakes only; the array owns its memory, so a consumer could re-enable writes.
                image.flags.writeable = False
                with self.lock:
                    if self.stop_event.is_set(): break  # never publish a frame after stop was requested
                    self.sequence += 1
                    if self.packet is not None: self.dropped += 1
                    self.frame_times.append(mono)
                    if time.monotonic() - start > self.config.slow_interval: self.slow_reads += 1
                    self.packet = FramePacket(image, self.sequence, redact_source(self.config.source), wall, mono, w, h, presentation_ms=pts, session_id=self.session_id)
                    self._latest = self.packet
                    self.condition.notify_all()
                if self.config.kind == 'file' and self.config.pace_video:
                    fps = self.actual.get('fps', 0)
                    if fps and 0 < fps < 1000:
                        # Absolute schedule: timer overshoot on one frame is absorbed by the next wait.
                        period, now = 1 / fps, time.monotonic()
                        next_due = start + period if next_due is None else next_due + period
                        if next_due < now - period: next_due = now  # fell behind; resync instead of bursting
                        self.stop_event.wait(max(0, next_due - now))
        except Exception as exc:
            diagnostic = redact_source(str(exc))
            log.error('Capture worker failed: %s', diagnostic)
            with self.lock:
                self.state = 'error'
                self.last_error = f'capture worker failed: {diagnostic}'
                self.packet = None
                self._latest = None
                self.condition.notify_all()
        finally:
            release_error = None
            capture, self.capture = self.capture, None
            if capture is not None:
                try:
                    capture.release()
                except Exception as exc:
                    release_error = redact_source(str(exc))
                    log.error('Capture release failed: %s', release_error)
            with self.lock:
                if release_error is not None:
                    prior_error = self.last_error if self.state == 'error' else None
                    self.state = 'error'
                    self.last_error = ((prior_error + '; ') if prior_error else '') + f'capture release failed: {release_error}'
                    self.packet = None
                    self._latest = None
                elif self.stop_event.is_set() and self.state == 'stopping':
                    self.state = 'stopped'
                if self.stop_event.is_set():
                    self.packet = None
                    self._latest = None
                self.condition.notify_all()

    def read(self):
        """Consume the latest packet without blocking; intermediate frames are discarded."""
        with self.lock:
            packet, self.packet = self.packet, None
            return packet

    def latest(self):
        """Return the shared latest packet without consuming it."""
        with self.lock:
            return self._latest

    def wait_for_frame(self, after_frame_id: int = 0, timeout: float | None = None, *, session_id: str | None = None):
        """Wait for a newer packet; raise TimeoutError, EOFError or CameraError on termination.

        Pass the previous packet's session_id to detect a transition even if it
        happened before this call. Without it, the current session is observed.
        """
        if not isinstance(after_frame_id, int) or isinstance(after_frame_id, bool) or after_frame_id < 0:
            raise ValueError('after_frame_id must be a nonnegative integer')
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 <= timeout < float('inf')):
            raise ValueError('timeout must be finite and nonnegative')
        if session_id is not None and not isinstance(session_id, str):
            raise ValueError('session_id must be a string or None')
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.condition:
            expected = self.session_id if session_id is None else session_id
            while True:
                if self.session_id != expected:
                    raise CameraError('Capture session changed')
                if self.state in ('stopped', 'stopping'):
                    raise CameraError('Capture stopped')
                if self.state == 'error':
                    raise CameraError(self.last_error or 'Capture failed')
                packet = self._latest
                if packet is not None and packet.frame_id > after_frame_id:
                    return packet
                if self.state == 'eof':
                    raise EOFError('End of video')
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError('Timed out waiting for a frame')
                self.condition.wait(remaining)

    def _rolling_fps(self, now_ns: int):
        """Frames per second over the last FPS_WINDOW_NS, measured up to now, so it decays during a stall."""
        times = self.frame_times
        if len(times) < 2: return None
        window_start = max(now_ns - FPS_WINDOW_NS, times[0])
        span = now_ns - window_start
        return sum(1 for t in times if t > window_start) / (span / 1e9) if span > 0 else None

    def status(self):
        with self.lock:
            now = time.monotonic_ns()
            times = list(self.frame_times)
            last = times[-1] if times else None
            return {'state': self.state, 'session_id': self.session_id, 'source': redact_source(self.config.source), 'actual': dict(self.actual),
                    'frames': self.sequence, 'failures': self.failures, 'dropped': self.dropped, 'slow_reads': self.slow_reads,
                    'rolling_fps': self._rolling_fps(now),
                    'frame_intervals_seconds': tuple((b - a) / 1e9 for a, b in zip(times, times[1:])),
                    'last_frame_monotonic_ns': last,
                    'seconds_since_last_frame': (now - last) / 1e9 if last is not None else None,
                    'last_error': self.last_error}

    def stop(self, timeout: float = 5):
        with self.lifecycle_lock:
            self._stop(timeout)

    def _stop(self, timeout: float = 5):
        self.stop_event.set()
        t = self.thread
        if t and t is not threading.current_thread():
            with self.lock:
                self.packet = None  # never hand out a pre-stop frame, even if the join below times out
                self._latest = None
                if t.is_alive() and self.state in ('running', 'starting'): self.state = 'stopping'
                self.condition.notify_all()
            t.join(timeout)
            if t.is_alive(): raise CameraError('Capture thread did not stop; backend read may be blocked')
        with self.lock:
            self.packet = None
            self._latest = None
            self.state = 'stopped'
            self.frame_times.clear()
            self.condition.notify_all()

    release = stop

    def switch_source(self, config: CameraConfig):
        if not isinstance(config, CameraConfig):
            raise TypeError('config must be a CameraConfig')
        with self.lifecycle_lock:
            self._stop()
            with self.lock:
                self.config = config
                self.actual = {}
            self._start()  # failed open leaves the old source released

    def __enter__(self): self.start(); return self
    def __exit__(self, *args): self.stop()
