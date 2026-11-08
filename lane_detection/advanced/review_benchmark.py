"""Comparable clean/clutter benchmark against an optional saved M03 baseline."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

from .adversarial import clutter_scene
from .synthetic import bev_fixture, scene


def load_engine(baseline: Path | None = None):
    if baseline:
        spec = importlib.util.spec_from_file_location("m03_saved_baseline", baseline / "__init__.py",
                                                     submodule_search_locations=[str(baseline)])
        engine = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = engine
        spec.loader.exec_module(engine)
    else:
        from lane_detection import advanced as engine
    return engine


def run(frames: int, warmup: int, baseline: Path | None = None, disable_hough: bool = False) -> dict:
    engine = load_engine(baseline)
    report = {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
              "opencv": cv2.__version__, "numpy": np.__version__, "opencv_threads": cv2.getNumThreads(),
              "baseline_directory": str(baseline) if baseline else None, "frames": frames, "warmup": warmup,
              "disable_hough": disable_hough,
              "acquisition_display_encoding_included": False, "cases": []}
    for w, h in ((320, 240), (640, 480)):
        for name in ("clean", "clutter"):
            image = clutter_scene(w, h) if name == "clutter" else scene(
                ((0, 0, .22 * w), (0, 0, .63 * w)), width=w, height=h)
            base = bev_fixture(image)
            kwargs = {"min_lane_width_px": 45 * w / 320, "max_lane_width_px": 260 * w / 320}
            if "max_frame_age_s" in {f.name for f in fields(engine.LaneConfig)}:
                kwargs["max_frame_age_s"] = None  # explicitly offline benchmark
            if disable_hough:
                kwargs["use_hough"] = False
            detector = engine.LaneDetector(engine.LaneConfig(**kwargs))
            stages, latency, states = [], [], {"valid": 0, "uncertain": 0, "invalid": 0}
            for i in range(warmup + frames):
                frame = replace(base, source_frame_id=i + 1, bev_sequence=i + 1,
                                acquired_monotonic_ns=time.monotonic_ns())
                start = time.perf_counter()
                result = detector.process(frame)
                elapsed = (time.perf_counter() - start) * 1000
                if i >= warmup:
                    latency.append(elapsed)
                    stages.append(asdict(result.timings))
                    states[result.validity] += 1
            report["cases"].append({"name": name, "size": [w, h], "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                                    "median_ms": float(np.median(latency)), "p95_ms": float(np.percentile(latency, 95)),
                                    "worst_ms": float(np.max(latency)), "throughput_fps": 1000 / float(np.mean(latency)),
                                    "validity_counts": states,
                                    "stages_median_ms": {key: float(np.median([s[key] for s in stages])) for key in stages[0]}})
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir", type=Path)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--disable-hough", action="store_true")
    args = p.parse_args(argv)
    if args.frames < 1 or args.warmup < 0:
        p.error("frames must be positive and warmup nonnegative")
    cv2.setNumThreads(1)
    report = run(args.frames, args.warmup, args.baseline_dir, args.disable_hough)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for case in report["cases"]:
        print(case["name"], case["size"], case["median_ms"], case["p95_ms"], case["worst_ms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
