"""Review availability/latency evaluation on identical seeded images."""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

from .availability_cases import availability_frame, clutter_case


def engine_at(directory: Path | None):
    if directory is None:
        from lane_detection import advanced
        return advanced
    name = "m03_availability_" + hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, directory / "__init__.py", submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def config_for(engine, frame, scale):
    if hasattr(engine.LaneConfig, "for_bev"):
        return engine.LaneConfig.for_bev(frame.bev, (.36, .68), max_frame_age_s=None)
    # Earlier pre-corrective comparison uses equivalent public pixel settings.
    args = dict(min_lane_width_px=90 * scale, max_lane_width_px=170 * scale, row_step=3 * scale,
                max_marking_width_px=20 * scale, max_gap_px=65 * scale, min_span_px=65 * scale,
                min_pair_overlap_px=65 * scale, association_margin_px=9 * scale,
                max_residual_px=3 * scale, inlier_threshold_px=4 * scale)
    return engine.LaneConfig(**{k:v for k,v in args.items() if k in {f.name for f in fields(engine.LaneConfig)}})


def memory_report(directory: Path | None, frames: int) -> list[dict]:
    """Memory measurements are separate from latency; RSS is process-wide."""
    import gc
    import os
    import subprocess
    import tracemalloc
    from dataclasses import asdict, replace
    engine = engine_at(directory)

    def rss():
        if sys.platform != "win32":
            return None
        command = f"Get-Process -Id {os.getpid()} | Select-Object WorkingSet64,PeakWorkingSet64 | ConvertTo-Json -Compress"
        return json.loads(subprocess.check_output(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                          creationflags=subprocess.CREATE_NO_WINDOW, text=True))

    output = []
    for scale in (1, 2):
        frame = availability_frame(clutter_case("specks", 200, 0, scale), scale)
        detector = engine.LaneDetector(config_for(engine, frame, scale))
        before_rss = rss()
        tracemalloc.start()
        try:
            for i in range(10):
                result = detector.process(replace(frame, source_frame_id=i + 1, bev_sequence=i + 1))
            gc.collect()
            initial = tracemalloc.get_traced_memory()[0]
            for i in range(10, 10 + frames):
                result = detector.process(replace(frame, source_frame_id=i + 1, bev_sequence=i + 1))
            gc.collect()
            live, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        diagnostics = getattr(result, "extraction_diagnostics", None)
        output.append(dict(size=[320*scale,240*scale],warmup=10,frames=frames,
            python_live_after_warmup_bytes=initial,python_live_final_bytes=live,
            python_live_growth_bytes=live-initial,python_peak_bytes=peak,
            process_memory_before=before_rss,process_memory_after=rss(),
            extraction=asdict(diagnostics) if diagnostics is not None else None))
    return output


def run(directory: Path | None = None, repeats: int = 3, stress: bool = False, *,
        use_hough: bool = True, geometry: str = "straight") -> dict:
    if geometry not in ("straight", "curve", "dashed", "competing"):
        raise ValueError("unknown evaluation geometry")
    engine = engine_at(directory)
    records = []
    settings = [("fragments", 0), ("fragments", 60), ("specks", 200)]
    if stress:
        settings = [("fragments", n) for n in (0, 30, 60, 120, 240)] + [("specks", n) for n in (100, 200, 400, 800)]
    from dataclasses import asdict, replace
    for scale in (1, 2):
        for kind, count in settings:
            for seed in range(8):
                image = clutter_case(kind, count, seed, scale, geometry)
                frame = availability_frame(image, scale)
                cfg = replace(config_for(engine, frame, scale), use_hough=use_hough)
                detector = engine.LaneDetector(cfg)
                durations = []
                stages = []
                last = None
                for i in range(repeats + 1):
                    packet = replace(frame, source_frame_id=i + 1, bev_sequence=i + 1, acquired_monotonic_ns=time.monotonic_ns())
                    start = time.perf_counter()
                    last = detector.process(packet)
                    if i:
                        durations.append((time.perf_counter() - start) * 1000)
                        stages.append(asdict(last.timings))
                error = None
                span = None
                if last.valid:
                    curve = last.selected_pair.centreline
                    rows = np.linspace(curve.v_min, curve.v_max, 100)
                    truth = 135 + (.001 * (rows / scale - 120) ** 2 if geometry == "curve" else 0)
                    error = float(np.max(np.abs(curve.evaluate(rows) / scale - truth)))
                    span = (curve.v_max - curve.v_min) / scale
                extra = getattr(last, "extraction_diagnostics", None)
                records.append(dict(kind=kind, count=count, seed=seed, size=[320*scale,240*scale],
                    image_sha256=hashlib.sha256(image.tobytes()).hexdigest(), valid=last.valid,
                    correct_valid=bool(last.valid and geometry != "competing" and error < 2.5),
                    incorrect_valid=bool(last.valid and (geometry == "competing" or error >= 2.5)),
                    centre_error_base_px=error, support_span_base_px=span, reasons=last.reason_codes,
                    median_ms=float(np.median(durations)), p95_ms=float(np.percentile(durations,95)), worst_ms=float(max(durations)),
                    latencies_ms=durations,
                    stage_median_ms={key: float(np.median([s[key] for s in stages])) for key in stages[0]},
                    budget_exhausted="processing_budget_exceeded" in last.reason_codes,
                    extraction=asdict(extra) if extra is not None else None))
    groups=[]
    for scale in (1,2):
        for kind,count in settings:
            rr=[r for r in records if r['size'][0]==320*scale and r['kind']==kind and r['count']==count]
            groups.append(dict(size=[320*scale,240*scale],kind=kind,count=count,scenes=len(rr),
                correct_valid=sum(r['correct_valid'] for r in rr), incorrect_valid=sum(r['incorrect_valid'] for r in rr),
                budget_exhaustions=sum(r['budget_exhausted'] for r in rr),
                median_ms=float(np.median([v for r in rr for v in r['latencies_ms']])),
                p95_ms=float(np.percentile([v for r in rr for v in r['latencies_ms']],95)),
                p95_scene_median_ms=float(np.percentile([r['median_ms'] for r in rr],95)),
                worst_ms=max(r['worst_ms'] for r in rr)))
    return dict(baseline_directory=str(directory) if directory else None, python=platform.python_version(),
                platform=platform.platform(), processor=platform.processor(), numpy=np.__version__, opencv=cv2.__version__,
                opencv_threads=cv2.getNumThreads(), repeats=repeats,warmup_per_scene=1,use_hough=use_hough,
                includes_acquisition_visualization=False,geometry=geometry,records=records,groups=groups)


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--stress",action="store_true")
    p.add_argument("--memory-frames",type=int,default=0)
    p.add_argument("--no-hough",action="store_true")
    p.add_argument("--geometry",choices=("straight","curve","dashed","competing"),default="straight")
    args=p.parse_args(argv)
    if args.repeats<1 or args.memory_frames<0:
        p.error("repeats must be positive")
    cv2.setNumThreads(1)
    report=run(args.baseline_dir,args.repeats,args.stress,use_hough=not args.no_hough,geometry=args.geometry)
    if args.memory_frames:
        report["memory"] = memory_report(args.baseline_dir,args.memory_frames)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for group in report['groups']:
        print(group)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
