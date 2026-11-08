"""Independent P1 acceptance matrix, optionally against the saved initial source."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from .adversarial import arc_scene, circle_u, edge_scene
from .review_benchmark import load_engine
from .synthetic import bev_fixture, scene


def evaluate(baseline: Path | None = None) -> dict:
    engine = load_engine(baseline)
    options = {"max_frame_age_s": None} if "max_frame_age_s" in {f.name for f in fields(engine.LaneConfig)} else {}
    report = {"baseline_directory": str(baseline) if baseline else None, "synthetic_only": True,
              "edge_cases": [], "arc_cases": [], "companion_cases": []}
    for mirrored in (False, True):
        for slope in (.15, .25, .4, .6, .8):
            for exit_row in (60, 100, 140):
                image, truth = edge_scene(slope, exit_row, mirrored)
                observation = engine.LaneDetector(engine.LaneConfig(**options)).process(bev_fixture(image))
                expected = not (slope == .15 and exit_row == 140)
                error = None
                if observation.valid:
                    pair = observation.selected_pair
                    by_id = {b.candidate_id: b for b in observation.boundaries}
                    error = max(float(np.max(np.abs(by_id[i].polynomial.evaluate(np.linspace(by_id[i].polynomial.v_min,
                        by_id[i].polynomial.v_max, 100)) - np.polyval(abc, np.linspace(by_id[i].polynomial.v_min,
                        by_id[i].polynomial.v_max, 100))))) for i, abc in zip((pair.left_id, pair.right_id), truth))
                report["edge_cases"].append({"mirrored": mirrored, "slope": slope, "exit_row": exit_row,
                    "expected_valid": expected, "valid": observation.valid, "max_boundary_error_px": error,
                    "reasons": observation.reason_codes,
                    "passed": observation.valid == expected and (error is None or error < 1.5)})
    for radius in (220, 300, 400, 700, 1200):
        for width in (60, 100, 140):
            for orientation in (-.35, 0.0, .35):
                for span in ((35, 205), (65, 175)):
                    image, circle = arc_scene(radius, width, orientation, span)
                    cfg = engine.LaneConfig(min_lane_width_px=width - 3, max_lane_width_px=width + 3,
                                            max_residual_px=.75, inlier_threshold_px=1.0, **options)
                    observation = engine.LaneDetector(cfg).process(bev_fixture(image))
                    centre_error, boundary_error, width_error = None, None, None
                    if observation.valid:
                        pair = observation.selected_pair
                        rows = np.linspace(pair.centreline.v_min, pair.centreline.v_max, 100)
                        centre_error = float(np.max(np.abs(pair.centreline.evaluate(rows) - circle_u(rows, circle))))
                        by_id = {b.candidate_id: b for b in observation.boundaries}
                        errors = []
                        for identifier, offset in ((pair.left_id, width / 2), (pair.right_id, -width / 2)):
                            poly = by_id[identifier].polynomial
                            rows = np.linspace(poly.v_min, poly.v_max, 100)
                            errors.append(float(np.max(np.abs(poly.evaluate(rows) - circle_u(rows, circle, offset)))))
                        boundary_error, width_error = max(errors), abs(pair.width_mean_px - width)
                    # Anchor moderate arcs to actual detection success. Tight/steep
                    # cases may be unsupported, but never silently inaccurate-valid.
                    cx, cy, _, _ = circle
                    radii = (radius + width / 2, radius - width / 2)
                    lower = max((span[0] - cy) / r for r in radii)
                    upper = min((span[1] - cy) / r for r in radii)
                    truth_normal_span = max(0.0, radius * (min(1, upper) - max(-1, lower)))
                    # Known angular overlap, including the fit's cap safety margin.
                    required = radius >= 400 and truth_normal_span >= cfg.min_pair_overlap_px + 9
                    passed = (observation.valid or not required) and (not observation.valid or
                              max(centre_error, boundary_error, width_error) < 1.5)
                    report["arc_cases"].append({"radius_px": radius, "width_px": width, "orientation": orientation,
                        "span": span, "required_valid": required, "valid": observation.valid,
                        "truth_normal_support_span_px": truth_normal_span,
                        "boundary_error_px": boundary_error, "centre_error_px": centre_error, "width_error_px": width_error,
                        "reasons": observation.reason_codes, "passed": passed})
    for separation in (10, 14, 18, 24, 32, 45):
        for u in (70 + separation, 200 - separation, 70 - separation, 200 + separation):
            observation = engine.LaneDetector(engine.LaneConfig(**options)).process(
                bev_fixture(scene(tuple((0, 0, x) for x in sorted((70, 200, u))))))
            error = abs(float(observation.selected_pair.centreline.evaluate(120)) - 135) if observation.valid else None
            report["companion_cases"].append({"separation_px": separation, "companion_u": u,
                "valid": observation.valid, "centre_error_px": error, "reasons": observation.reason_codes,
                "passed": not observation.valid and observation.selected_pair is None})
    report["summary"] = {name: {"count": len(report[name]), "passed": sum(c["passed"] for c in report[name]),
                                   "valid": sum(c["valid"] for c in report[name])}
                         for name in ("edge_cases", "arc_cases", "companion_cases")}
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    report = evaluate(args.baseline_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return int(any(s["passed"] < s["count"] for s in report["summary"].values()) and args.baseline_dir is None)


if __name__ == "__main__":
    raise SystemExit(main())
