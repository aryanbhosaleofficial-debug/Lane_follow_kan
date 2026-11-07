"""Runtime M02 pipeline: M01 frames in, metric bird's-eye-view frames out.

M02 never opens a camera. It consumes :class:`camera_input.FramePacket` objects (duck-typed: any
object with the same attributes works) that M01's ``CameraManager`` already publishes, using only
M01's non-consuming interface (``wait_for_frame`` / ``latest``), so it cannot take frames away from
other consumers.

Frame identity contract for downstream modules
----------------------------------------------
Each :class:`BEVFrame` carries the identity of the frame it was computed from -- ``(source_session_id,
source_frame_id)`` -- plus M01's acquisition timestamps unchanged. ``bev_sequence`` increases only
when a NEW source frame has been processed; processing the same source frame again returns the
cached object, so equal ``bev_sequence`` means "same BEV". ``age_seconds()`` measures from M01's
*acquisition* time, i.e. it is end-to-end software frame age, not sensor-exposure-to-output latency.
"""
from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from ._util import is_positive_finite, redact_credentials
from .errors import (CalibrationError, FrameUnavailableError, InvalidCalibrationInputError, InvalidFrameError,
                     ProfileError, ResolutionMismatchError, StaleFrameError)
from .mapper import BEVMapper, CameraGeometry
from .profile import CalibrationProfile
from .bev import BEVConfig
from .status import CalibrationStatus
from .validation import ValidationReport

log = logging.getLogger(__name__)

_REQUIRED_PACKET_FIELDS = ("image", "frame_id", "source_id", "acquired_unix_ns", "acquired_monotonic_ns", "width", "height")


class FrameLike(Protocol):
    """The slice of M01's ``FramePacket`` that M02 reads."""
    image: Any
    frame_id: int
    source_id: str
    acquired_unix_ns: int
    acquired_monotonic_ns: int
    width: int
    height: int
    pixel_format: str
    valid: bool
    session_id: str


@dataclass(frozen=True, eq=False)
class BEVFrame:
    """One bird's-eye-view result with full provenance. Arrays are read-only and may be shared."""
    image: np.ndarray                       # (H, W, 3) uint8 BGR; meaningful only where valid_mask == 255
    valid_mask: np.ndarray                  # (H, W) uint8: 255 where the camera observed the ground point
    support_mask: np.ndarray                # (H, W) uint8: observed AND inside the calibrated region (+ margin)
    geometry: CameraGeometry                # coordinate conversions for this frame's grid
    bev: BEVConfig
    bev_sequence: int
    source_frame_id: int
    source_session_id: str
    source_id: str
    acquired_unix_ns: int
    acquired_monotonic_ns: int
    source_width: int
    source_height: int
    processed_monotonic_ns: int
    processing_seconds: float
    profile_fingerprint: str
    source_frames_skipped: int | None = None    # frames M01 published between the previous processed frame and this one (same session)
    source_presentation_ms: float | None = None
    undistorted_image: np.ndarray | None = None
    notes: tuple[str, ...] = ()
    pixel_format: str = "BGR"
    calibration_status: CalibrationStatus = CalibrationStatus.METRIC_UNVERIFIED

    def validated_geometry(self) -> CameraGeometry:
        """Metric geometry only after independent measured validation was supplied."""
        if self.calibration_status is not CalibrationStatus.VALID_METRIC:
            raise ProfileError("metric distances are not validated for this frame; supply held-out physical measurements")
        return self.geometry

    @property
    def key(self) -> tuple[str, int]:
        """Unambiguous identity of the source frame: (session id, frame id)."""
        return (self.source_session_id, self.source_frame_id)

    @property
    def meters_per_pixel(self) -> float:
        return self.bev.meters_per_pixel

    def age_seconds(self) -> float:
        """Seconds since M01 acquired the source frame (software acquisition time)."""
        return max(0.0, (time.monotonic_ns() - self.acquired_monotonic_ns) / 1e9)


@dataclass(frozen=True)
class ProcessorConfig:
    """Runtime policy.

    ``max_frame_age_s``: frames older than this (since M01 acquisition) are refused as stale;
    ``None`` disables the check. ``strict_source``: enforce the profile's ``source_ids`` binding.
    ``allow_uniform_rescale``: accept frames at a different resolution of the same aspect ratio by
    rescaling the profile -- only valid if the camera delivers the same field of view at that size;
    off by default because many cameras crop in other modes.
    """
    max_frame_age_s: float | None = 0.5
    strict_source: bool = True
    include_undistorted: bool = False
    allow_uniform_rescale: bool = False

    def __post_init__(self):
        if self.max_frame_age_s is not None and not is_positive_finite(self.max_frame_age_s):
            raise InvalidCalibrationInputError("max_frame_age_s must be None or positive and finite")
        for name in ("strict_source", "include_undistorted", "allow_uniform_rescale"):
            if not isinstance(getattr(self, name), bool):
                raise InvalidCalibrationInputError(f"{name} must be a boolean")


class BEVProcessor:
    """Turns M01 frame packets into :class:`BEVFrame` results. Thread-safe (calls are serialised)."""

    def __init__(self, profile: CalibrationProfile, config: ProcessorConfig | None = None,
                 validation_report: ValidationReport | None = None):
        profile.require_metric()
        if validation_report is not None and (not isinstance(validation_report, ValidationReport)
                or validation_report.evidence != "measured" or not validation_report.results
                or not validation_report.passed or validation_report.profile_fingerprint != profile.fingerprint()):
            raise ProfileError("VALID_METRIC needs a passing measured validation report for this exact profile")
        self.profile = profile
        self.calibration_status = (CalibrationStatus.VALID_METRIC if validation_report is not None
                                   else CalibrationStatus.METRIC_UNVERIFIED)
        self.config = config or ProcessorConfig()
        self._mapper = BEVMapper(profile)               # built now: the first frame pays no setup cost
        self._rescaled: dict[tuple[int, int], BEVMapper] = {}
        self._fingerprint = profile.fingerprint()
        self._lock = threading.RLock()
        self._sequence = 0
        self._last: BEVFrame | None = None
        self._last_frame_id_by_session: tuple[str, int] | None = None
        self._last_source_id: str | None = None
        self._counters = {"processed": 0, "repeats": 0, "stale": 0, "session_changes": 0, "source_changes": 0, "errors": 0}
        self._errors_by_type: dict[str, int] = {}
        self._durations: deque[float] = deque(maxlen=200)

    # -- main entry --------------------------------------------------------------------------
    def process(self, packet: FrameLike) -> BEVFrame:
        """Compute the BEV of ``packet``. Raises a :class:`CalibrationError` subclass on any problem; never returns an
        uncalibrated or partially valid result. The packet's image is not modified."""
        with self._lock:
            try:
                return self._process(packet)
            except Exception as exc:
                self._counters["errors"] += 1
                name = type(exc).__name__
                self._errors_by_type[name] = self._errors_by_type.get(name, 0) + 1
                raise

    def _process(self, packet: FrameLike) -> BEVFrame:
        image, w, h = self._validate_packet(packet)
        key = (str(packet.session_id), int(packet.frame_id))
        last = self._last
        if last is not None and last.key == key:
            self._counters["repeats"] += 1
            return last                                     # same source frame: do not claim a new BEV
        cfg = self.config
        if cfg.max_frame_age_s is not None:
            age = (time.monotonic_ns() - int(packet.acquired_monotonic_ns)) / 1e9
            if age > cfg.max_frame_age_s:
                self._counters["stale"] += 1
                raise StaleFrameError(f"frame {key[1]} is {age:.3f} s old (limit {cfg.max_frame_age_s} s)")
        source_id = redact_credentials(packet.source_id)       # M01 already redacts; never rely on it for logs / errors
        notes: list[str] = []
        mapper = self._mapper_for(w, h, source_id, notes)

        previous_session = self._last_frame_id_by_session
        if previous_session is not None and previous_session[0] != key[0]:
            self._counters["session_changes"] += 1
            log.info("capture session changed (source %s)", source_id)
        if self._last_source_id is not None and source_id != self._last_source_id:
            self._counters["source_changes"] += 1
            log.info("source changed to %s", source_id)
        skipped = None
        if previous_session is not None and previous_session[0] == key[0]:
            skipped = max(0, key[1] - previous_session[1] - 1)

        t0 = time.perf_counter()
        try:
            bev_image = mapper.warp(image)
            undistorted = mapper.undistorter().undistort(image) if cfg.include_undistorted else None
        except cv2.error as exc:
            raise InvalidFrameError(f"OpenCV failed while processing frame {key[1]}: {str(exc).splitlines()[0][:160]}") from None
        bev_image.flags.writeable = False
        if undistorted is not None:
            undistorted.flags.writeable = False
        duration = time.perf_counter() - t0

        self._sequence += 1
        frame = BEVFrame(
            image=bev_image, valid_mask=mapper.valid_mask, support_mask=mapper.support_mask, geometry=mapper.geometry,
            bev=mapper.bev, bev_sequence=self._sequence, source_frame_id=key[1], source_session_id=key[0], source_id=source_id,
            acquired_unix_ns=int(packet.acquired_unix_ns), acquired_monotonic_ns=int(packet.acquired_monotonic_ns),
            source_width=w, source_height=h, processed_monotonic_ns=time.monotonic_ns(), processing_seconds=duration,
            profile_fingerprint=self._fingerprint, source_frames_skipped=skipped,
            source_presentation_ms=getattr(packet, "presentation_ms", None), undistorted_image=undistorted, notes=tuple(notes),
            calibration_status=self.calibration_status)
        self._last = frame
        self._last_frame_id_by_session = key
        self._last_source_id = source_id
        self._counters["processed"] += 1
        self._durations.append(duration)
        return frame

    def process_latest(self, camera: Any) -> BEVFrame:
        """Process M01's newest published frame without consuming it. Repeated calls on the same frame return the cached BEV."""
        packet = camera.latest()
        if packet is None:
            raise FrameUnavailableError("the camera has not published a frame (stopped, starting, or failed)", reason="no_frame")
        return self.process(packet)

    # -- validation helpers ----------------------------------------------------------------
    @staticmethod
    def _validate_packet(packet: Any) -> tuple[np.ndarray, int, int]:
        if packet is None:
            raise InvalidFrameError("no frame packet supplied")
        missing = [f for f in _REQUIRED_PACKET_FIELDS if not hasattr(packet, f)]
        if missing:
            raise InvalidFrameError(f"packet lacks required fields {missing}; expected an M01 FramePacket")
        if getattr(packet, "valid", True) is not True:
            raise InvalidFrameError("packet is marked invalid by the source")
        fmt = getattr(packet, "pixel_format", "BGR")
        if fmt != "BGR":
            raise InvalidFrameError(f"unsupported pixel_format {fmt!r}; M02 expects BGR")
        image = packet.image
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise InvalidFrameError(f"image must be a non-empty HxWx3 uint8 array, got {getattr(image, 'shape', type(image).__name__)} "
                                    f"{getattr(image, 'dtype', '')}")
        h, w = image.shape[:2]
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0
               for v in (packet.width, packet.height)):
            raise InvalidFrameError("packet.width and packet.height must be positive integers")
        if (packet.width, packet.height) != (w, h):
            raise InvalidFrameError(f"packet metadata {packet.width}x{packet.height} disagrees with the image {w}x{h}")
        for name in ("frame_id", "acquired_unix_ns", "acquired_monotonic_ns"):
            v = getattr(packet, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0:
                raise InvalidFrameError(f"packet.{name} must be a non-negative integer, got {v!r}")
        if not isinstance(getattr(packet, "session_id", ""), str) or not isinstance(packet.source_id, str):
            raise InvalidFrameError("packet.session_id and packet.source_id must be strings")
        return image, w, h

    def _mapper_for(self, w: int, h: int, source_id: str, notes: list[str]) -> BEVMapper:
        base = self.profile
        if (w, h) == base.image_size:
            base.check_compatibility(w, h, source_id, strict_source=self.config.strict_source)
            return self._mapper
        if not self.config.allow_uniform_rescale:
            base.check_compatibility(w, h, source_id, strict_source=self.config.strict_source)     # raises the specific error
        mapper = self._rescaled.get((w, h))
        if mapper is None:
            try:
                scaled = base.scaled_to(w, h)
            except (InvalidCalibrationInputError, ProfileError, ValueError) as exc:
                raise ResolutionMismatchError(f"frame is {w}x{h}, profile {base.image_size[0]}x{base.image_size[1]}; cannot rescale: {exc}") from None
            mapper = BEVMapper(scaled)
            if len(self._rescaled) >= 2:
                self._rescaled.pop(next(iter(self._rescaled)))
            self._rescaled[(w, h)] = mapper
        mapper.profile.check_compatibility(w, h, source_id, strict_source=self.config.strict_source)
        notes.append(f"profile rescaled from {base.image_size[0]}x{base.image_size[1]} to {w}x{h}; valid only if the camera delivers the same field of view")
        return mapper

    # -- diagnostics -----------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        with self._lock:
            d = list(self._durations)
            return {**self._counters, "errors_by_type": dict(self._errors_by_type), "last_bev_sequence": self._sequence,
                    "last_processing_ms": d[-1] * 1e3 if d else None,
                    "mean_processing_ms": statistics.fmean(d) * 1e3 if d else None,
                    "max_processing_ms": max(d) * 1e3 if d else None}


class ManagerFrameSource:
    """Reads M01 frames for ONE consumer through ``wait_for_frame`` (non-consuming, no polling loop).

    It tracks the last frame id and session it returned, so it never hands the same frame twice and
    survives M01 source switches. M01 raises ``CameraError("Capture stopped")`` for a moment while it
    switches source, which is indistinguishable from a real stop; this class tells them apart through
    ``camera.status()``: a terminal ``error`` state is reported, anything else is retried (with a short
    sleep, so it never spins) until the timeout.
    """

    def __init__(self, camera: Any, *, retry_interval_s: float = 0.01):
        if not is_positive_finite(retry_interval_s):
            raise InvalidCalibrationInputError("retry_interval_s must be positive and finite")
        for method in ("wait_for_frame", "status"):
            if not callable(getattr(camera, method, None)):
                raise InvalidCalibrationInputError(f"camera must provide {method}(); expected an M01 CameraManager")
        self.camera = camera
        self.retry_interval_s = retry_interval_s
        self._last_id = 0
        self._session: str | None = None

    def next_frame(self, timeout: float) -> Any:
        """Return the next frame newer than the last one returned. Raises ``FrameUnavailableError`` (``.reason`` is
        ``timeout``, ``camera_error`` or ``stopped``) or ``EOFError`` at the end of a recording."""
        if not is_positive_finite(timeout):
            raise InvalidCalibrationInputError("timeout must be positive and finite")
        deadline = time.monotonic() + timeout
        last_state = "unknown"
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reason = "stopped" if last_state in ("stopped", "stopping") else "timeout"
                raise FrameUnavailableError(f"no new frame within {timeout} s (camera state: {last_state})", reason=reason)
            try:
                packet = self.camera.wait_for_frame(self._last_id, timeout=remaining, session_id=self._session)
            except TimeoutError:
                last_state = self._state()
                continue
            except EOFError:
                raise
            except RuntimeError as exc:                       # M01 CameraError subclasses RuntimeError
                status = self.camera.status()
                last_state = status.get("state", "unknown")
                if last_state == "error":
                    raise FrameUnavailableError(f"camera failed: {status.get('last_error') or exc}", reason="camera_error") from exc
                self._session = None                          # session changed / switching: observe whatever comes next
                time.sleep(min(self.retry_interval_s, max(0.0, deadline - time.monotonic())))
                continue
            self._last_id = int(packet.frame_id)
            self._session = str(packet.session_id)
            return packet

    def _state(self) -> str:
        try:
            return self.camera.status().get("state", "unknown")
        except Exception:                                      # status() must never break frame delivery
            return "unknown"


class BEVWorker:
    """Background thread that turns M01 frames into BEV frames ONCE and shares the latest with many consumers.

    Use it when several perception modules need the same BEV: they all read ``latest()`` /
    ``wait_for_bev()`` instead of each repeating the warp. Only one BEV (the newest) is retained, so
    memory is bounded and a slow consumer never delays the worker or its peers.

    States: ``stopped``, ``running``, ``eof`` (recording finished; the final BEV stays available),
    ``error`` (camera failed). A per-frame processing problem (stale frame, resolution/source
    mismatch, invalid image) does NOT stop the worker: the previous BEV is withdrawn (it must not
    pass for current), the error is recorded and delivered to waiting consumers, and the next frame
    is tried.
    """

    LOG_EVERY = 100          # repeat the warning for a persistent fault only every this many occurrences

    def __init__(self, processor: BEVProcessor, source: Any, *, wait_timeout_s: float = 0.25):
        if not is_positive_finite(wait_timeout_s):
            raise InvalidCalibrationInputError("wait_timeout_s must be positive and finite")
        if not hasattr(source, "next_frame"):
            source = ManagerFrameSource(source)               # accept a CameraManager directly
        self.processor = processor
        self.source = source
        self._wait_timeout = wait_timeout_s
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle = threading.Lock()
        self._latest: BEVFrame | None = None
        self._state = "stopped"
        self._last_error: str | None = None
        self._error_serial = 0
        self._last_exception: Exception | None = None
        self._error_counts: dict[str, int] = {}
        self._counters = {"published": 0, "stale_skipped": 0, "processing_errors": 0, "timeouts": 0}

    # -- lifecycle --------------------------------------------------------------------------
    def start(self) -> None:
        with self._lifecycle:
            if self._thread is not None and self._thread.is_alive():
                if self._stop.is_set():
                    raise CalibrationError("worker is still stopping; call stop() again")
                return
            self._stop.clear()
            with self._cond:
                self._latest = None
                self._state = "running"
                self._last_error = None
                self._cond.notify_all()
            self._thread = threading.Thread(target=self._run, name="bev-worker", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the thread and wait up to ``timeout``. Waiting consumers are woken. Raises ``CalibrationError`` if the
        thread is still blocked (e.g. inside the camera call) after the timeout; call ``stop()`` again later."""
        with self._lifecycle:
            self._stop.set()
            with self._cond:
                self._latest = None
                self._cond.notify_all()
            t = self._thread
            if t is not None and t is not threading.current_thread():
                t.join(timeout)
                if t.is_alive():
                    raise CalibrationError("BEV worker did not stop in time (blocked in the frame source?)")
            with self._cond:
                self._state = "stopped"
                self._cond.notify_all()

    def __enter__(self) -> "BEVWorker":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- worker loop ------------------------------------------------------------------------
    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    packet = self.source.next_frame(self._wait_timeout)
                except FrameUnavailableError as exc:
                    if exc.reason == "camera_error":
                        self._finish("error", str(exc))
                        return
                    self._counters["timeouts"] += 1
                    continue
                except EOFError:
                    self._finish("eof", None)
                    return
                if self._stop.is_set():
                    return
                try:
                    frame = self.processor.process(packet)
                except StaleFrameError as exc:
                    self._counters["stale_skipped"] += 1
                    self._record_error(exc, withdraw=True)
                    continue
                except CalibrationError as exc:
                    self._counters["processing_errors"] += 1
                    self._record_error(exc, withdraw=True)
                    continue
                with self._cond:
                    if self._stop.is_set():
                        return
                    self._latest = frame
                    self._last_error = None
                    self._counters["published"] += 1
                    self._cond.notify_all()
        except Exception as exc:                                # defensive: a bug must surface, not hang consumers
            log.exception("BEV worker crashed")
            self._finish("error", f"worker crashed: {exc}")

    def _record_error(self, exc: Exception, *, withdraw: bool) -> None:
        text = f"{type(exc).__name__}: {exc}"
        # Rate-limit by error TYPE (messages embed changing frame numbers/ages): log the first occurrence, then every
        # LOG_EVERY-th, so a persistent fault at 30 fps cannot flood the log.
        kind = type(exc).__name__
        count = self._error_counts.get(kind, 0) + 1
        self._error_counts[kind] = count
        if count == 1 or count % self.LOG_EVERY == 0:
            log.warning("BEV frame rejected (%d so far for %s): %s", count, kind, text)
        with self._cond:
            if withdraw:
                self._latest = None
            self._last_error = text
            self._last_exception = exc
            self._error_serial += 1
            self._cond.notify_all()

    def _finish(self, state: str, error: str | None) -> None:
        with self._cond:
            self._state = state
            if error is not None:
                self._last_error = error
                self._latest = None
            self._cond.notify_all()

    # -- consumer API -----------------------------------------------------------------------
    def latest(self) -> BEVFrame | None:
        """Newest BEV, or None if none is current (not started, withdrawn after an error, or stopped). Non-blocking."""
        with self._cond:
            return self._latest

    def wait_for_bev(self, after_sequence: int = 0, timeout: float | None = None) -> BEVFrame:
        """Block until a BEV with ``bev_sequence > after_sequence`` exists. Each consumer keeps its own ``after_sequence``.

        Raises ``TimeoutError``; ``EOFError`` when a recording ended with nothing newer; ``FrameUnavailableError`` if the
        worker stopped or the camera failed; or the processing error (a :class:`CalibrationError`) if a frame that arrived
        while waiting was rejected.
        """
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < 0:
            raise InvalidCalibrationInputError("after_sequence must be a non-negative integer")
        if timeout is not None and not (isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and 0 <= timeout < float("inf")):
            raise InvalidCalibrationInputError("timeout must be None or finite and non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            serial = self._error_serial
            while True:
                if self._latest is not None and self._latest.bev_sequence > after_sequence:
                    return self._latest
                if self._error_serial != serial and self._last_exception is not None:
                    raise self._last_exception
                if self._state == "error":
                    raise FrameUnavailableError(self._last_error or "camera failed", reason="camera_error")
                if self._state == "eof":
                    raise EOFError("end of video")
                if self._state == "stopped":
                    raise FrameUnavailableError("BEV worker is stopped", reason="stopped")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out waiting for a BEV frame")
                self._cond.wait(remaining)

    def status(self) -> dict[str, Any]:
        with self._cond:
            return {"state": self._state, "last_error": self._last_error, "latest_sequence": None if self._latest is None else self._latest.bev_sequence,
                    "latest_age_seconds": None if self._latest is None else self._latest.age_seconds(), **self._counters,
                    "processor": self.processor.stats()}
