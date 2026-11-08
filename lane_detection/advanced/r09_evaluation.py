"""Exact reviewer clutter, identical-input before/after timing and Hough ablation."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from .availability_cases import availability_frame
from .availability_evaluation import engine_at, memory_report
from .r09_cases import review_case


def run(directory=None, repeats=10, *, hough=True, breadth=False, stress=False):
    engine = engine_at(directory)
    records = []
    settings = [("fragments", n) for n in (0, 40, 60, 80, 100)] + [("specks", n) for n in (100, 200, 300)]
    if stress:
        settings += [("fragments", n) for n in (150,200,300,400)] + [("specks",n) for n in (500,800,1200)]
    geometries = ("straight", "curve", "dashed", "multi", "companion", "no_lane", "noise") if breadth else ("straight",)
    for scale in (1, 2):
        for geometry in geometries:
            for kind, count in settings:
                for seed in range(8):
                    image = review_case(kind, count, seed, scale, geometry)
                    frame = availability_frame(image, scale)
                    # Preserve p9_r09a's default width bounds, not the narrower
                    # author's availability benchmark (.36,.68) metre assumption.
                    cfg = engine.LaneConfig.for_bev(frame.bev, (.18, 1.04), max_frame_age_s=None, use_hough=hough)
                    detector = engine.LaneDetector(cfg)
                    durations = []
                    for i in range(repeats + 2):
                        packet = replace(frame, source_frame_id=i + 1, bev_sequence=i + 1)
                        start = time.perf_counter()
                        result = detector.process(packet)
                        if i >= 2:
                            durations.append((time.perf_counter() - start) * 1000)
                    errors = []
                    midpoint = None
                    if result.selected_pair is not None:
                        by_id = {b.candidate_id: b for b in result.boundaries}
                        for bid, c in ((result.selected_pair.left_id, 70), (result.selected_pair.right_id, 200)):
                            p = by_id[bid].polynomial
                            rows = np.linspace(p.v_min, p.v_max, 100)
                            truth = c + (.001 * (rows / scale - 120) ** 2 if geometry == "curve" else 0)
                            errors.append(float(np.max(np.abs(p.evaluate(rows) / scale - truth))))
                        midpoint = all(abs(float(by_id[bid].polynomial.evaluate(120 * scale)) / scale - c) < 2
                                       for bid, c in ((result.selected_pair.left_id,70),(result.selected_pair.right_id,200)))
                    expected = geometry in ("straight", "curve", "dashed")
                    correct = bool(result.valid and expected and max(errors) < 2.5)
                    records.append(dict(scale=scale, geometry=geometry, kind=kind, count=count, seed=seed,
                        image_sha256=hashlib.sha256(image.tobytes()).hexdigest(), valid=result.valid,
                        correct_valid=correct, incorrect_valid=bool(result.valid and not correct),
                        original_midpoint_check=midpoint, max_boundary_error_base_px=max(errors) if errors else None,
                        limited_support=result.limited_support, support=result.usable_support_v_px,
                        confidence=result.confidence, reasons=result.reason_codes, latencies_ms=durations,
                        extraction=asdict(result.extraction_diagnostics)))
    groups = []
    for scale in (1, 2):
        for geometry in geometries:
            for kind, count in settings:
                rr = [r for r in records if (r['scale'],r['geometry'],r['kind'],r['count']) == (scale,geometry,kind,count)]
                times = [t for r in rr for t in r['latencies_ms']]
                groups.append(dict(scale=scale, geometry=geometry, kind=kind, count=count, scenes=len(rr),
                    correct_valid=sum(r['correct_valid'] for r in rr), incorrect_valid=sum(r['incorrect_valid'] for r in rr),
                    original_midpoint_pass=sum(r['valid'] and r['original_midpoint_check'] is True for r in rr),
                    budget_events=sum(r['extraction']['budget_reason'] is not None for r in rr),
                    comparisons_max=max(r['extraction']['association_comparisons'] for r in rr),
                    active_tracks_max=max(r['extraction']['peak_active_tracks'] for r in rr),
                    median_ms=float(np.median(times)), p95_ms=float(np.percentile(times,95)), max_ms=max(times)))
    return dict(python=platform.python_version(),platform=platform.platform(),processor=platform.processor(),
                opencv=cv2.__version__,numpy=np.__version__,threads=cv2.getNumThreads(),
                baseline_directory=str(directory) if directory else None,repeats=repeats,warmup=2,hough=hough,
                records=records,groups=groups)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--breadth',action='store_true')
    p.add_argument('--stress',action='store_true')
    p.add_argument('--no-hough',action='store_true')
    p.add_argument('--memory-frames',type=int,default=0)
    a=p.parse_args()
    if a.repeats < 1 or a.memory_frames < 0:
        p.error('repeats must be positive; memory-frames nonnegative')
    cv2.setNumThreads(1)
    result=run(a.baseline_dir,a.repeats,hough=not a.no_hough,breadth=a.breadth,stress=a.stress)
    if a.memory_frames:
        result['memory']=memory_report(a.baseline_dir,a.memory_frames)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for g in result['groups']:
        print(g,flush=True)


if __name__=='__main__':
    main()
