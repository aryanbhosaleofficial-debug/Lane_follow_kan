"""Measured CPU benchmark. Synthetic workloads, no acquisition/display/encoding."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from camera_calibration import BEVConfig, BEVProcessor, ProcessorConfig
from camera_calibration.benchmark import build_case, _packet
from . import LaneConfig, LaneDetector
from .synthetic import bev_fixture, scene


def _summary(values: list[float]) -> dict[str, float]:
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "worst_ms": float(np.max(values))}


def run_benchmark(frames: int = 200, warmup: int = 20) -> dict:
    if frames < 1 or warmup < 0:
        raise ValueError("frames must be positive and warmup nonnegative")
    report = {"hardware": {"platform": platform.platform(), "processor": platform.processor(),
                          "python": platform.python_version(), "opencv": cv2.__version__,
                          "numpy": np.__version__, "opencv_threads": cv2.getNumThreads()},
              "synthetic": True, "raspberry_pi_measurement": False,
              "includes_acquisition_display_encoding": False, "debug": False,
              "frames_per_case": frames, "warmup_per_case": warmup, "cases": []}
    for width, height in ((320, 240), (640, 480)):
        scale = width / 320
        cfg = LaneConfig(min_lane_width_px=45 * scale, max_lane_width_px=260 * scale, max_frame_age_s=None)
        left = (.001 / scale, -.24, 84.4 * scale)
        right = (.001 / scale, -.24, 184.4 * scale)
        for name in ("solid", "curved_dashed_noise", "m02_plus_m03"):
            setup = time.perf_counter()
            processor = None
            if name == "m02_plus_m03":
                profile, raw = build_case(width, height, mpp=2 / height)
                bev = BEVConfig(width, height, 2 / height, width / 2 - .5, 2.3 / (2 / height) - .5)
                processor = BEVProcessor(profile.with_bev(bev), ProcessorConfig(max_frame_age_s=None))
                # Different BEV grid: use the actual ~0.5m synthetic road's pixel width.
                detector = LaneDetector(LaneConfig(min_lane_width_px=20, max_lane_width_px=180, max_frame_age_s=None))
            else:
                image = scene(((0, 0, 70 * scale), (0, 0, 170 * scale)) if name == "solid" else (left, right),
                              width=width, height=height, dashed=name != "solid", noise=name != "solid")
                base = bev_fixture(image)
                detector = LaneDetector(cfg)
            setup_ms = (time.perf_counter() - setup) * 1e3
            stages: dict[str, list[float]] = {key: [] for key in ("preprocessing_ms", "extraction_ms", "fitting_ms", "association_ms", "total_ms")}
            latencies = []
            valid = 0
            validity_counts = {"valid": 0, "uncertain": 0, "invalid": 0}
            for i in range(warmup + frames):
                packet = _packet(raw, i + 1) if processor else None
                frame = None if processor else replace(base, source_frame_id=i + 1, bev_sequence=i + 1)
                start = time.perf_counter()
                if processor:
                    frame = processor.process(packet)
                observation = detector.process(frame)
                elapsed_ms = (time.perf_counter() - start) * 1e3
                if i >= warmup:
                    valid += observation.valid
                    validity_counts[observation.validity] += 1
                    latencies.append(elapsed_ms)
                    for key, value in asdict(observation.timings).items():
                        stages[key].append(value)
            report["cases"].append({"name": name, "source_size": [width, height] if processor else None,
                                    "bev_size": [width, height], "setup_ms": setup_ms,
                                    "wall_latency": _summary(latencies),
                                    "throughput_fps": 1000 / float(np.mean(latencies)),
                                    "valid_observations": valid,
                                    "validity_counts": validity_counts,
                                    "workload_note": ("M02 synthetic road includes a centre dashed marking; multiple candidates, no selected lane expected"
                                                      if processor else "independent synthetic pixel-polynomial truth"),
                                    "m03_stages": {key: _summary(values) for key, values in stages.items()}})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.frames < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("frames/threads must be positive; warmup must be nonnegative")
    cv2.setNumThreads(args.threads)
    report = run_benchmark(args.frames, args.warmup)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
