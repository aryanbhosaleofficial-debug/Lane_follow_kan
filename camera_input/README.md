# M01 camera input

Install from the repository root: `python -m pip install -r requirements.txt`. Run the tests with `python -m unittest camera_input.test_camera -v`.

Manual webcam: `python -m camera_input.demo --source 0`. Recorded video: `python -m camera_input.demo --source videos/test.mp4 --file`. Press q to stop. Add `--fast` for unpaced video. Switch after five seconds with `--switch-to 1` or `--switch-to file:videos/test.mp4`.

```python
import time
from camera_input import CameraConfig, CameraManager
with CameraManager(CameraConfig(source=0, width=640, height=480, fps=30)) as camera:
    while camera.status()['state'] == 'running':
        packet = camera.read()
        if packet is None:
            time.sleep(0.005)
            continue
        image = packet.image  # OpenCV BGR; treat as read-only
        age = packet.age_seconds()
        # Pass packet to the next module.
```

`read()` consumes the most recent frame without waiting. The one-slot buffer replaces unread frames and counts replacements as dropped. A consumer owns the returned array reference; it remains valid after subsequent reads and is not copied, so consumers should not mutate an image shared with others. IDs increase across restarts and source switches. Timestamps are software acquisition times after `read()` returns: Unix nanoseconds and monotonic nanoseconds, not sensor exposure times. File `presentation_ms` is the backend-reported position in milliseconds, which can be unavailable or inaccurate. FPS is rolling acquisition throughput, not sensor FPS; video pacing uses nominal file FPS, not presentation timestamps. Actual property values are backend-reported and may be zero or inaccurate. Resize is optional and changes packet dimensions. Status reports bounded interval history, failures, slow reads and drops.

A switch stops and releases the old source before opening the new one; if opening fails, the manager enters error state and the old source is not restored. The worker exclusively owns the capture; stop waits for it and raises if the backend blocks longer than the timeout. Some OpenCV backends may block indefinitely on network reads; backend-specific timeouts or process isolation are needed for a hard shutdown guarantee. A file that ends before its container-reported frame count (beyond a 1%/2-frame tolerance), has no readable frames, or yields frames that are not HxWx3 uint8 ends in `error` with `last_error` set rather than `eof`; files whose container frame count is only an estimate (some variable-FPS files) may be reported as premature. Camera permissions, codecs, frame buffering, timestamps and actual properties depend on OS and OpenCV backend. Test on the target vehicle before relying on frame freshness. Lifecycle operations are serialized. If stop times out, the state becomes `stopping`, the buffered packet is discarded, and the worker retains ownership of its capture until the backend read returns; `start()` and `switch_source()` raise `CameraError` until then. Call stop again after it unblocks. A caller must not assume the device was released after a timeout. Opening a source does not hold the lock used by `read()` and `status()`, so consumers keep running (and see `starting`) while a slow device or stream opens.

**EOF completeness tolerance.** Because OpenCV's `CAP_PROP_FRAME_COUNT` can be a container estimate, a file that ends within `max(2, 1%)` of the reported count is treated as a clean `eof`; a larger shortfall is a terminal `error` (`premature end of file`). So a clean `eof` on a 590-frame file means at least 585 frames were decoded, not necessarily all 590. A consumer that needs an exact count must compare `status()['frames']` with its own independent count (M02's video export does).


## Configuration and status

Set `kind="file"` for a local recorded file; live camera indices and OpenCV-supported stream URLs use `kind="live"`. Resolution and FPS are requests for live sources, and the backend may ignore them. `resize=(width, height)` explicitly transforms output. `retries` bounds consecutive live reconnection attempts, with exponential delays bounded by `max_retry_delay`; stopping interrupts the delay. Recorded files default to nominal-FPS pacing; `pace_video=False` decodes as quickly as possible.

`status()` returns `stopped`, `starting`, `running`, `stopping`, `eof`, or `error` (a switch passes through `stopped` and `starting`), along with negotiated properties and cumulative counters. `last_frame_monotonic_ns` and `seconds_since_last_frame` expose camera stalls: a live source can stay `running` while a backend read is blocked, so health checks should use `seconds_since_last_frame`, not state alone. `rolling_fps` counts frames over the last second up to now, so it falls during a stall. `last_error` holds the most recent failure reason and survives `stop()`. Clean EOF does not count as a failure. Paced video follows an absolute schedule at the nominal file FPS. URL credentials (`user:pass@`) are replaced by `***@` in `status()['source']`, `FramePacket.source_id`, exceptions and log messages; OpenCV/FFmpeg's own native logging is outside this module's control. An unread final video frame remains available after EOF. On a live read failure the buffered packet is cleared before retrying. `dropped` counts unread packet replacements, not frames discarded by the camera or backend. Rolling throughput and frame intervals reset on stop or switch; cumulative counts and frame IDs persist for the manager lifetime. After a failed switch, call `switch_source` or `start` again to recover.

The demo can accept a stream URL as `--source`; avoid exposing credentials in command history or logs. Backends may block in opening or reading a stream, and cannot necessarily be interrupted from Python. No guarantee is made about RTSP support, frame age at acquisition, exposure timestamps, delivery rate, or hardware performance.


## Multiple consumers and waiting

`latest()` returns the shared most recently published `FramePacket` without consuming it. `read()` retains its legacy independent consume-on-read mailbox semantics; calling it does not remove `latest()`. Both reference the same packet and underlying image, with no copies. Published images are marked read-only (`writeable=False`). NumPy rejects writes, but some OpenCV 4.x drawing functions ignore this flag and can corrupt shared input. Always draw on `packet.image.copy()`. This guards against mistakes, not deliberate misuse: the array owns its memory, so a consumer could re-enable writes. A consumer that needs to modify an image must use `packet.image.copy()` (measured at about 0.3 ms for 864x496 on the review machine). Packets already returned remain valid after a switch; they belong to their original session and are not current frames. A successful switch clears the shared slot before the new source opens.

Each successful or attempted new capture session gets a unique `session_id`, stable through automatic reconnects. Frame IDs are monotonic for a manager's lifetime. To wait efficiently, use `packet = camera.wait_for_frame(last_id, timeout=1.0, session_id=last_session)` and update both values for each consumer independently. Omit `session_id` on the first call to observe the current session. A changed session or stopped/failed capture raises `CameraError`; timeout raises `TimeoutError`, and completed video with no newer frame raises `EOFError`. A final packet may be returned once at EOF if its ID is newer. A switch can wake an existing waiter with a stopped error before the next session starts; obtain the new session and wait again. `latest()` is nonblocking and may return the final file frame at EOF; check state and age for freshness.

`dropped` counts unread replacements in the legacy `read()` mailbox, not a per-consumer loss count. Since `latest()` does not consume that mailbox, this count can grow even when latest-only consumers are keeping up. No per-consumer queues or delivery guarantees are provided.

Only H×W×3 `uint8` BGR frames are supported. A nonempty unsupported format terminates the session with a diagnostic; it is not silently converted or retried. `CameraConfig(source="0")` selects index 0 for a live source; file-kind numeric strings are file paths. Resize must be a tuple of two positive integer dimensions. Optional `backend` is an integer OpenCV backend constant, such as `cv2.CAP_DSHOW`, `cv2.CAP_V4L2`, or `cv2.CAP_GSTREAMER`; support depends on the installed build and platform. The default `None` preserves OpenCV's default selection. `backoff_factor` controls exponential retry growth; `retries` and `max_retry_delay` bound recovery. An `error` is terminal until the caller explicitly restarts or switches. Calling `start()` after `error` or `eof` waits up to 5 s for the previous worker to finish releasing its source before starting a new session, and raises `CameraError` if it has not finished. The initial `start()` does not retry a failed open; supervisors should retry at boot if the camera may enumerate late.

## Target hardware verification — NOT RUN

On the Raspberry Pi 5 or deployment computer, with the vehicle safely stationary and authorization to use the camera:

- Check negotiated resolution and backend-reported FPS against the requested configuration.
- Measure acquisition throughput, frame age and stall metrics under representative lighting and processing load.
- Measure CPU and memory use over an extended run; verify bounded memory and acceptable temperature.
- Check camera startup, shutdown, unplug/replug, recovery limits, and supervisor restart.
- Run multiple consumers simultaneously and verify shared frame IDs, session changes, and source switching.
- Test actual streams and codecs if used, including interruption and recovery; record backend timeouts and shutdown behavior.

These checks do not establish sensor exposure-to-processing latency unless independently instrumented.

A backend exception during capture release is recorded in `last_error` and wakes frame waiters. The device may not have been released even if a later `stop()` sets state to `stopped`.
