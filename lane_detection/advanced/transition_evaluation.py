"""Physical transition availability and scoped-geometry ground truth."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .availability_cases import availability_frame, transition_case
from .availability_evaluation import config_for, engine_at


def run(directory: Path | None = None) -> dict:
    engine = engine_at(directory)
    cases = []
    for scale in (1, 2):
        for radius in (220, 300, 700):
            for transition in (90, 120, 150):
                for direction in (1, -1):
                    for dashed in (False, True):
                        image = transition_case(radius, transition, direction=direction, dashed=dashed, scale=scale)
                        near = image.copy()
                        near[:transition * scale] = 35
                        f = availability_frame(image, scale)
                        cfg = config_for(engine, f, scale)
                        near_result = engine.LaneDetector(cfg).process(availability_frame(near, scale))
                        start = time.perf_counter()
                        r = engine.LaneDetector(cfg).process(f)
                        elapsed = (time.perf_counter() - start) * 1000
                        error = None
                        if r.valid:
                            p = r.selected_pair.centreline
                            rows = np.linspace(p.v_min, p.v_max, 100) / scale
                            delta = np.minimum(rows - transition, 0)
                            truth = 135 + radius - np.sqrt(radius ** 2 - delta ** 2)
                            if direction == -1:
                                truth = 319 - truth
                            # Nearest-neighbour scaling shifts pixel-centre raster
                            # evidence by up to (scale-1)/2; allow it in base units.
                            error = float(np.max(np.abs(p.evaluate(rows * scale) / scale - truth)))
                        success = (r.valid or not near_result.valid) and (not r.valid or error < 2.0)
                        cases.append(dict(size=[320*scale,240*scale],radius_base_px=radius,transition_base_row=transition,
                            direction=direction,dashed=dashed, image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                            near_control_valid=near_result.valid,valid=r.valid,limited=getattr(r,'limited_support',False),
                            support=getattr(r,'usable_support_v_px',None),confidence=r.confidence,centre_error_base_px=error,
                            reasons=r.reason_codes,latency_ms=elapsed,passed=success))
    return dict(baseline_directory=str(directory) if directory else None,cases=cases,
                passed=sum(c['passed'] for c in cases),failed=sum(not c['passed'] for c in cases),
                required=sum(c['near_control_valid'] for c in cases),
                required_recovered=sum(c['near_control_valid'] and c['valid'] for c in cases),
                valid=sum(c['valid'] for c in cases),limited=sum(c['limited'] for c in cases))


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    report=run(args.baseline_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print({k:v for k,v in report.items() if k!='cases'})
    return int(report['failed']>0 and args.baseline_dir is None)


if __name__=='__main__':
    raise SystemExit(main())
