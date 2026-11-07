"""Measure M02 runtime performance on THIS machine:  python -m camera_calibration.benchmark [--frames N]

Everything is measured, nothing is assumed. Read the output as results for the machine and OpenCV
build printed in the header, on SYNTHETIC frames. They are NOT Raspberry Pi 5 numbers and NOT camera
numbers (no sensor, USB or driver time is included); repeat on the target hardware before relying on them.
"""
from __future__ import annotations

import argparse
import gc
import platform
import statistics
import time
import tracemalloc

import cv2
import numpy as np

from . import mapper as mapper_mod
from .bev import BEVConfig
from .pipeline import BEVProcessor, ProcessorConfig
from .plane_calibration import RoadPlaneCalibrator, RoadPlaneConfig
from .profile import new_profile
from .synthetic import SyntheticCamera, SyntheticPose, make_road_scene, project_ground_points

_GROUND = np.column_stack([m.ravel() for m in np.meshgrid(np.linspace(0.5, 1.6, 4), np.linspace(-0.35, 0.35, 3))])


def build_case(width: int, height: int, mpp: float = 0.004, interpolation: str = "linear", fixed_point: bool = False):
    """A synthetic camera at ``width x height`` (focal length scaled with width), its profile and a rendered road frame."""
    s = width / 640.0
    cam = SyntheticCamera(width=width, height=height, fx=520.0 * s, fy=520.0 * s, cx=(width - 1) / 2, cy=(height - 1) / 2,
                          dist=(-0.28, 0.09, 0.001, -0.0008, -0.01))
    pose = SyntheticPose(0.25, 28.0)
    prof = new_profile(cam.intrinsics(), "bench")
    fit = RoadPlaneCalibrator(prof.intrinsics, prof.undistortion, RoadPlaneConfig()).estimate(project_ground_points(cam, pose, _GROUND), _GROUND)
    bev = BEVConfig.from_extents(0.3, 2.3, -0.7, 0.7, mpp, interpolation=interpolation, fixed_point_maps=fixed_point)
    profile = prof.with_road_plane(fit).with_bev(bev)
    return profile, make_road_scene("straight").render(cam, pose)


def _packet(img: np.ndarray, frame_id: int):
    from types import SimpleNamespace
    img.flags.writeable = False
    return SimpleNamespace(image=img, frame_id=frame_id, source_id="bench", acquired_unix_ns=time.time_ns(),
                           acquired_monotonic_ns=time.monotonic_ns(), width=img.shape[1], height=img.shape[0], session_id="bench")


def time_processor(profile, frame: np.ndarray, frames: int, warmup: int = 20) -> dict[str, float]:
    """First-frame vs warm per-frame latency of the whole processor (cache cleared first), in milliseconds."""
    mapper_mod.clear_map_cache()
    t0 = time.perf_counter()
    proc = BEVProcessor(profile, ProcessorConfig(max_frame_age_s=None))        # map build happens here, once
    setup_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    proc.process(_packet(frame, 1))
    first_ms = (time.perf_counter() - t0) * 1e3
    for i in range(warmup):
        proc.process(_packet(frame, 2 + i))
    lat = []
    for i in range(frames):
        pkt = _packet(frame, 1000 + i)
        t0 = time.perf_counter()
        proc.process(pkt)
        lat.append((time.perf_counter() - t0) * 1e3)
    lat.sort()
    return {"setup_ms": setup_ms, "first_frame_ms": first_ms, "mean_ms": statistics.fmean(lat), "p50_ms": lat[len(lat) // 2],
            "p95_ms": lat[int(len(lat) * 0.95)], "max_ms": lat[-1], "fps_equivalent": 1000.0 / statistics.fmean(lat)}


def memory_growth(profile, frame: np.ndarray, frames: int = 300) -> float:
    """Net Python-allocation growth (bytes) across ``frames`` processed frames while keeping only the latest result."""
    proc = BEVProcessor(profile, ProcessorConfig(max_frame_age_s=None))
    for i in range(20):
        proc.process(_packet(frame, i + 1))
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    for i in range(frames):
        proc.process(_packet(frame, 100 + i))
    gc.collect()
    grown = tracemalloc.get_traced_memory()[0] - base
    tracemalloc.stop()
    return float(grown)


def end_to_end_age(profile, frame: np.ndarray, seconds: float = 3.0, source_fps: float = 30.0) -> dict[str, float]:
    """Frame age at the consumer through the REAL M01 CameraManager + BEVWorker, with a fake capture paced at ``source_fps``.

    Age is measured from M01's acquisition timestamp, i.e. software acquisition -> BEV available. A synthetic source has no
    sensor, USB or driver latency, so a real camera's age will be higher by at least its exposure + transfer + decode time.
    """
    from camera_input import CameraConfig, CameraManager
    from .pipeline import BEVWorker

    class PacedCapture:
        def __init__(self, source): self.next = time.perf_counter()
        def isOpened(self): return True
        def set(self, *a): return True
        def get(self, prop): return source_fps
        def release(self): pass
        def read(self):
            self.next += 1.0 / source_fps
            wait = self.next - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            return True, frame.copy()

    ages, skipped = [], 0
    cam = CameraManager(CameraConfig(source=0), PacedCapture)
    worker = BEVWorker(BEVProcessor(profile, ProcessorConfig(max_frame_age_s=None)), cam)
    cam.start(); worker.start()
    try:
        after, end = 0, time.monotonic() + seconds
        while time.monotonic() < end:
            bev = worker.wait_for_bev(after, timeout=1.0)
            ages.append(bev.age_seconds() * 1e3)
            skipped += bev.source_frames_skipped or 0
            after = bev.bev_sequence
    finally:
        worker.stop(); cam.stop()
    ages.sort()
    return {"frames": float(len(ages)), "skipped": float(skipped), "p50_ms": ages[len(ages) // 2], "p95_ms": ages[int(len(ages) * 0.95)], "max_ms": ages[-1]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args(argv)
    print(f"machine: {platform.platform()} | {platform.processor() or 'unknown CPU'} | python {platform.python_version()} | OpenCV {cv2.__version__} "
          f"| cv2 threads {cv2.getNumThreads()} | SYNTHETIC frames, no camera, not a Raspberry Pi 5\n")
    cases = [("640x480", 640, 480), ("1280x720", 1280, 720), ("1920x1080", 1920, 1080)]
    print(f"{'source':<10} {'BEV px':<9} {'interp':<7} {'maps':<6} {'setup ms':>9} {'first ms':>9} {'mean ms':>8} {'p50':>7} {'p95':>7} {'max':>7} {'fps-equiv':>10}")
    for name, w, h in cases:
        for interp, fixed in (("linear", False), ("linear", True), ("nearest", False), ("cubic", False)):
            if (w, h) != (640, 480) and (interp, fixed) != ("linear", False):
                continue                                              # other resolutions: default configuration only
            profile, frame = build_case(w, h, interpolation=interp, fixed_point=fixed)
            r = time_processor(profile, frame, args.frames)
            print(f"{name:<10} {profile.bev.width_px}x{profile.bev.height_px:<4} {interp:<7} {'int16' if fixed else 'f32':<6} {r['setup_ms']:>9.1f} {r['first_frame_ms']:>9.2f} "
                  f"{r['mean_ms']:>8.2f} {r['p50_ms']:>7.2f} {r['p95_ms']:>7.2f} {r['max_ms']:>7.2f} {r['fps_equivalent']:>10.0f}")
    print("\nBEV grid size at 640x480 source (linear, float32 maps = default):")
    for mpp in (0.008, 0.004, 0.002):
        profile, frame = build_case(640, 480, mpp=mpp)
        r = time_processor(profile, frame, args.frames)
        print(f"  {profile.bev.width_px}x{profile.bev.height_px} px ({mpp * 1000:.0f} mm/px): setup {r['setup_ms']:.0f} ms, mean {r['mean_ms']:.2f} ms, p95 {r['p95_ms']:.2f} ms")
    profile, frame = build_case(640, 480)
    print(f"\nmemory: net Python allocation growth over {args.frames} frames = {memory_growth(profile, frame, args.frames) / 1e6:.3f} MB "
          f"(one BEV is {profile.bev.width_px * profile.bev.height_px * 3 / 1e6:.2f} MB)")
    try:
        e = end_to_end_age(profile, frame)
        print(f"\nend-to-end through M01 CameraManager + BEVWorker (synthetic 30 fps source): {int(e['frames'])} BEVs, {int(e['skipped'])} source frames "
              f"skipped; frame age at consumer p50 {e['p50_ms']:.2f} ms, p95 {e['p95_ms']:.2f} ms, max {e['max_ms']:.2f} ms (software only: no sensor/USB/driver time)")
    except ImportError:
        print("\nend-to-end measurement skipped: camera_input (M01) is not importable from here")
    print("\nsingle-thread cv2 for comparison:")
    cv2.setNumThreads(1)
    r = time_processor(profile, frame, args.frames)
    print(f"  640x480 linear/float32: mean {r['mean_ms']:.2f} ms, p95 {r['p95_ms']:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
