"""Recorded-video validation through M01. Pixel demo is never metric calibration.

Two modes:
* ``--profile PATH``: frames go through M02's real runtime (``BEVProcessor``) with a metric profile.
* ``--quad ...``: an illustrative ``cv2.warpPerspective`` of a manually chosen image quadrilateral. M02's runtime is NOT
  used, nothing is undistorted, and the result is not a calibrated bird's-eye view.

A run either finishes completely or leaves ``RUN_INCOMPLETE.txt`` (with the error) in the output folder; metrics from an
earlier run are removed at the start so they can never vouch for new, partial outputs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np

from camera_input import CameraConfig, CameraManager, CameraError
from .pipeline import BEVProcessor, ProcessorConfig
from .profile import CalibrationProfile
from .status import CalibrationStatus
from . import diagnostics as diag

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
OUTPUT_VIDEOS = ("original_vs_bev.mp4", "bev_only.mp4", "bev_validity_mask.mp4", "mask_diagnostic.mp4")
INCOMPLETE_MARKER = "RUN_INCOMPLETE.txt"
SAMPLE_COUNT = 5
#: M01 cannot be started "paused", so a consumer that is descheduled for more than one frame period at start-up misses
#: frame 1. Nothing has been processed at that point, so the capture is restarted (bounded). Later losses fail at once.
START_ATTEMPTS = 3


class _FirstFramesLost(RuntimeError):
    """M01 replaced the first frame(s) before this consumer's first read (start-up race; safe to retry)."""


def inspect_video(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"video does not exist: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"unsupported video extension: {path.suffix}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"OpenCV cannot open video (codec or file problem): {path}")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, count = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            raise ValueError(f"video has no decodable first frame: {path}")
        if (frame.shape[1], frame.shape[0]) != (width, height):
            raise ValueError("backend frame dimensions disagree with video properties")
        return {"path": str(path.resolve()), "width": width, "height": height, "fps_reported": fps,
                "frames_reported": count, "duration_estimate_s": count / fps if fps > 0 else None}
    finally:
        cap.release()


def pixel_demo_matrix(quad: np.ndarray, width: int, height: int, output_size: tuple[int, int]) -> np.ndarray:
    """Map a manually selected image quadrilateral into an output rectangle, in pixels only."""
    pts = np.asarray(quad, dtype=np.float64)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise ValueError("quad must be four finite (x,y) pixel points")
    if np.unique(pts, axis=0).shape[0] != 4 or np.any(pts[:, 0] < 0) or np.any(pts[:, 0] >= width) or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] >= height):
        raise ValueError("quad has duplicate or out-of-frame points")
    area = cv2.contourArea(pts.astype(np.float32))
    if area < 0.01 * width * height or not cv2.isContourConvex(pts.astype(np.float32)):
        raise ValueError("quad is too small or not convex")
    ow, oh = output_size
    dst = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], np.float32)
    H = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("degenerate perspective transform")
    return H


def _writer(path: Path, fps: float, size: tuple[int, int], color: bool = True):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size, color)
    if not writer.isOpened():
        raise OSError(f"cannot create output video: {path}")
    return writer


def _panel(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Fit without stretching; black letterbox space is explicit."""
    width, height = size
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    canvas = np.zeros((height, width, 3), np.uint8)
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _verify_video(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"saved video cannot be reopened: {path}")
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if frame is None or frame.size == 0: raise OSError(f"invalid saved frame: {path}")
            n += 1
    finally:
        cap.release()
    return n


def _begin_run(output: Path) -> Path:
    """Mark the folder as in progress and remove artefacts that could vouch for this run's partial files."""
    (output / "processing_metrics.json").unlink(missing_ok=True)
    samples = output / "sample_frames"
    if samples.is_dir():
        for old in samples.glob("frame_*.jpg"):
            old.unlink()
    marker = output / INCOMPLETE_MARKER
    marker.write_text("This run has not finished successfully. Do not use the files in this folder.\n", encoding="utf-8")
    return marker


def _sample_positions(expected_frames: int) -> set[int]:
    """Zero-based processing positions to save as JPEG: evenly spread over the expected length (first and last included)."""
    if expected_frames <= 0:
        return {0}
    return {round(k * (expected_frames - 1) / (SAMPLE_COUNT - 1)) for k in range(SAMPLE_COUNT)}


def process_video(video: Path, output: Path, *, profile_path: Path | None = None,
                  quad: np.ndarray | None = None, output_size: tuple[int, int] = (640, 600),
                  limit: int | None = None, pace: bool = True) -> dict:
    info = inspect_video(video)
    if (len(output_size) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in output_size)):
        raise ValueError("output_size must contain two positive integers")
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer")
    if not pace:
        raise ValueError("full-frame validation requires pacing with M01's latest-frame buffer")
    map_start = time.perf_counter()
    H = pixel_mask = processor = None
    if profile_path is not None:
        profile = CalibrationProfile.load(profile_path)
        profile.require_metric()
        profile.check_compatibility(info["width"], info["height"], str(video.resolve()))
        processor = BEVProcessor(profile, ProcessorConfig(max_frame_age_s=None))
        status = "METRIC_UNVERIFIED - requires independent physical validation"
        engine = "M02 BEVProcessor (fused undistortion + road-plane remap)"
        output_size = (profile.bev.width_px, profile.bev.height_px)
    else:
        if quad is None:
            raise ValueError("pixel demonstration requires an explicit --quad; metric profile requires --profile")
        H = pixel_demo_matrix(quad, info["width"], info["height"], output_size)
        status = "PIXEL DEMO - NO METRIC SCALE OR UNDISTORTION"
        engine = "cv2.warpPerspective of a hand-picked quad (M02 runtime NOT used)"
        pixel_mask = cv2.warpPerspective(np.full((info["height"], info["width"]), 255, np.uint8), H, output_size, flags=cv2.INTER_NEAREST)
    map_initialization_ms = (time.perf_counter() - map_start) * 1000
    output.mkdir(parents=True, exist_ok=True)
    marker = _begin_run(output)
    try:
        for attempt in range(1, START_ATTEMPTS + 1):
            try:
                metrics = _run(video, output, info, processor, H, pixel_mask, quad, output_size, limit, pace, status, engine, map_initialization_ms)
                break
            except _FirstFramesLost as exc:
                if attempt == START_ATTEMPTS:
                    raise RuntimeError(f"{exc} (start-up race persisted for {START_ATTEMPTS} attempts)") from exc
        metrics["start_attempts"] = attempt
        (output / "processing_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except BaseException as exc:
        marker.write_text(f"This run FAILED and its outputs are incomplete. Do not use the files in this folder.\n"
                          f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise
    marker.unlink()
    return metrics


def _run(video, output, info, processor, H, pixel_mask, quad, output_size, limit, pace, status, engine, map_initialization_ms) -> dict:
    ow, oh = output_size
    fps = info["fps_reported"] if 0 < info["fps_reported"] <= 240 else 24.0
    expected = min(limit, info["frames_reported"]) if limit else info["frames_reported"]
    samples_at = _sample_positions(expected)
    writers = {}
    timings, encode_timings, wait_timings, frame_ids, ages, session_ids = [], [], [], [], [], set()
    decoded, first_ns = 0, time.perf_counter_ns()
    tracemalloc.start()
    try:
        writers["original_vs_bev.mp4"] = _writer(output / "original_vs_bev.mp4", fps, (2 * ow, oh))
        writers["bev_only.mp4"] = _writer(output / "bev_only.mp4", fps, (ow, oh))
        writers["bev_validity_mask.mp4"] = _writer(output / "bev_validity_mask.mp4", fps, (ow, oh), color=False)
        writers["mask_diagnostic.mp4"] = _writer(output / "mask_diagnostic.mp4", fps, (ow, oh))
        with (output / "frame_metadata.jsonl").open("w", encoding="utf-8") as metadata_file, \
                CameraManager(CameraConfig(source=str(video.resolve()), kind="file", pace_video=pace)) as camera:
            last_id, session = 0, None
            while limit is None or decoded < limit:
                try:
                    wait_start = time.perf_counter()
                    packet = camera.wait_for_frame(last_id, timeout=5.0, session_id=session)
                    wait_timings.append(time.perf_counter() - wait_start)
                except EOFError:
                    break
                except (TimeoutError, CameraError) as exc:
                    raise RuntimeError(f"M01 acquisition failed after {decoded} frames: {exc}") from exc
                # A fresh CameraManager numbers frames from 1; anything else means M01 replaced frames before we read them.
                expected_id = last_id + 1 if decoded else 1
                if packet.frame_id != expected_id:
                    lost = (_FirstFramesLost if decoded == 0 else RuntimeError)
                    raise lost(f"M01 published source frames {expected_id}..{packet.frame_id - 1} before this consumer read them; "
                               "output would be incomplete (re-run with less CPU load)")
                if packet.image.shape != (info["height"], info["width"], 3):
                    raise ValueError("video frame resolution changed during processing")
                t = time.perf_counter()
                if processor is not None:
                    bev = processor.process(packet)
                    image, mask = bev.image, bev.valid_mask
                else:
                    image = cv2.warpPerspective(packet.image, H, output_size)
                    mask = pixel_mask
                timings.append(time.perf_counter() - t)
                frame_ids.append(packet.frame_id)
                session_ids.add(packet.session_id)
                ages.append(packet.age_seconds())
                if processor is not None:
                    source_overlay = diag.draw_reference_points(packet.image, processor.profile)
                    bev_overlay = diag.draw_bev_grid(image, processor.profile.bev, spacing_m=0.25)
                else:
                    source_overlay = packet.image.copy()  # OpenCV drawing may ignore NumPy's read-only flag
                    cv2.polylines(source_overlay, [np.round(quad).astype(np.int32).reshape(-1, 1, 2)],
                                  True, (0, 0, 255), 2)
                    bev_overlay = image
                original = _panel(source_overlay, output_size)
                comparison = np.concatenate((original, bev_overlay), axis=1)
                cv2.putText(comparison, f"Original | frame {packet.frame_id}", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 1)
                title = "Metric BEV" if processor is not None else "Perspective demo"
                cv2.putText(comparison, f"{title} | {status}", (ow + 12, 25), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
                mask_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                coverage = float(np.count_nonzero(mask)) / mask.size
                cv2.putText(mask_view, f"Source coverage {coverage:.1%}", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1)
                cv2.putText(mask_view, "Ground support UNKNOWN" if processor is None else "Ground support: see profile", (12, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 255), 1)
                t = time.perf_counter()
                writers["original_vs_bev.mp4"].write(comparison)
                writers["bev_only.mp4"].write(image)
                writers["bev_validity_mask.mp4"].write(mask)
                writers["mask_diagnostic.mp4"].write(mask_view)
                encode_timings.append(time.perf_counter() - t)
                if decoded in samples_at:
                    samples = output / "sample_frames"; samples.mkdir(exist_ok=True)
                    if not cv2.imwrite(str(samples / f"frame_{packet.frame_id:04d}.jpg"), comparison):
                        raise OSError("could not save sample frame")
                last_id, session = packet.frame_id, packet.session_id
                metadata_file.write(json.dumps({"frame_id": packet.frame_id, "session_id": packet.session_id,
                                                "source_id": packet.source_id,
                                                "acquired_unix_ns": packet.acquired_unix_ns,
                                                "acquired_monotonic_ns": packet.acquired_monotonic_ns,
                                                "presentation_ms": packet.presentation_ms,
                                                "transform_ms": timings[-1] * 1000}) + "\n")
                decoded += 1
            state = camera.status()
            if limit is None and state["state"] != "eof":
                raise RuntimeError(f"processing stopped before clean EOF: {state['state']}: {state['last_error']}")
    finally:
        for writer in writers.values(): writer.release()
        current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    elapsed = (time.perf_counter_ns() - first_ns) / 1e9
    verified = {name: _verify_video(output / name) for name in writers}
    if any(n != decoded for n in verified.values()):
        raise RuntimeError(f"saved-video frame count mismatch: processed {decoded}, decoded {verified}")
    skipped = (frame_ids[0] - 1 if frame_ids else 0) + sum(b - a - 1 for a, b in zip(frame_ids, frame_ids[1:]))
    metrics = {"input": info, "calibration_status": status,
               "calibration_state": (processor.calibration_status.value if processor else CalibrationStatus.PIXEL_ONLY.value),
               "profile_fingerprint": processor.profile.fingerprint() if processor else None,
               "transform_engine": engine, "frames_processed": decoded,
               "first_frame_id": frame_ids[0] if frame_ids else None, "last_frame_id": frame_ids[-1] if frame_ids else None,
               "source_sessions": sorted(session_ids), "m01_dropped_mailbox_frames": state["dropped"],
               "source_frames_skipped": skipped, "m01_final_state": state["state"], "output_verified_frames": verified,
               "transform_mean_ms": statistics.mean(timings) * 1000 if timings else None,
               "transform_median_ms": statistics.median(timings) * 1000 if timings else None,
               "transform_p95_ms": sorted(timings)[int(.95 * (len(timings) - 1))] * 1000 if timings else None,
               "transform_max_ms": max(timings) * 1000 if timings else None,
               "map_initialization_ms": map_initialization_ms,
               "m01_wait_total_s": sum(wait_timings), "encode_total_s": sum(encode_timings), "pipeline_elapsed_s": elapsed,
               "pipeline_fps": decoded / elapsed if elapsed else None,
               "mean_software_frame_age_ms": statistics.mean(ages) * 1000 if ages else None,
               "python_peak_mb": peak / 1e6}
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", type=Path)
    ap.add_argument("--quad", type=float, nargs=8, metavar="PX", help="TL TR BR BL image pixels; illustrative pixel demo only")
    ap.add_argument("--size", type=int, nargs=2, default=(640, 600), metavar=("WIDTH", "HEIGHT"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    if (args.profile is None) == (args.quad is None):
        ap.error("specify exactly one of --profile or --quad")
    result = process_video(args.video, args.out, profile_path=args.profile,
                           quad=np.asarray(args.quad).reshape(4, 2) if args.quad else None,
                           output_size=tuple(args.size), limit=args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
