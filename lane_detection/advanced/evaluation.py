"""Reproducible synthetic evaluation/demo with independently known pixel truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import LaneConfig, LaneDetector
from .synthetic import bev_fixture, scene
from .visualization import draw_observation


def evaluate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    straight = ((0, 0, 70), (0, 0, 170))
    gentle = ((.001, -.24, 84.4), (.001, -.24, 184.4))
    tight = ((.003, -.72, 113.2), (.003, -.72, 213.2))
    junction = scene()
    cv2.line(junction, (170, 220), (290, 10), (255, 255, 255), 5)
    cases = [
        ("straight", scene(straight), straight, LaneConfig()),
        ("gentle_curve", scene(gentle), gentle, LaneConfig()),
        ("tight_curve", scene(tight), tight, LaneConfig()),
        ("dashed_curve", scene(gentle, dashed=True), gentle, LaneConfig()),
        ("broken_curve", scene(gentle, broken=True), gentle, LaneConfig()),
        ("shadow", scene(straight, shadow=True), straight, LaneConfig(preprocessing="local_contrast", white_min_value=90)),
        ("noise", scene(straight, noise=True), straight, LaneConfig()),
        ("dashed_curve_noise", scene(gentle, dashed=True, noise=True), gentle, LaneConfig()),
        ("one_boundary", scene(((0, 0, 70),)), None, LaneConfig()),
        ("empty", scene(()), None, LaneConfig()),
        ("noise_only", scene((), noise=True), None, LaneConfig()),
        ("multiple_lanes", scene(((0, 0, 50), (0, 0, 150), (0, 0, 250))), None, LaneConfig()),
        ("junction", junction, None, LaneConfig()),
    ]
    records, tiles = [], []
    for index, (name, image, truth, config) in enumerate(cases):
        detector = LaneDetector(config)
        observation = detector.process(bev_fixture(image, index + 1), debug=True)
        record = {"name": name, "expected_valid": truth is not None, "valid": observation.valid,
                  "validity": observation.validity, "confidence": observation.confidence,
                  "reason_codes": observation.reason_codes, "max_boundary_error_px": None,
                  "max_centreline_error_px": None}
        if observation.valid and truth is not None:
            pair = observation.selected_pair
            by_id = {b.candidate_id: b for b in observation.boundaries}
            rows = np.linspace(pair.centreline.v_min, pair.centreline.v_max, 100)
            errors = [np.max(np.abs(by_id[identifier].polynomial.evaluate(rows) - np.polyval(abc, rows)))
                      for identifier, abc in zip((pair.left_id, pair.right_id), truth)]
            record["max_boundary_error_px"] = float(max(errors))
            centre_truth = np.mean([np.polyval(abc, rows) for abc in truth], axis=0)
            record["max_centreline_error_px"] = float(np.max(np.abs(pair.centreline.evaluate(rows) - centre_truth)))
        record["passed"] = (observation.valid == record["expected_valid"]
                            and (record["max_boundary_error_px"] is None or record["max_boundary_error_px"] < 2.5))
        records.append(record)
        overlay = draw_observation(image, observation, detector.last_debug)
        cv2.putText(overlay, name, (5, 232), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        for label, artifact in (("overlay", overlay), ("candidates", detector.last_debug.candidate_mask),
                                ("accepted", detector.last_debug.accepted_mask)):
            if not cv2.imwrite(str(output / f"{name}_{label}.png"), artifact):
                raise OSError(f"cannot write {name} {label}")
        tiles.append(overlay)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    montage = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    if not cv2.imwrite(str(output / "montage.png"), montage):
        raise OSError("cannot write montage")
    report = {"synthetic_only": True, "metric_evidence": False, "cases": records,
              "passed": sum(r["passed"] for r in records), "failed": sum(not r["passed"] for r in records)}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/m03/synthetic"))
    args = parser.parse_args(argv)
    report = evaluate(args.output)
    print(json.dumps(report, indent=2))
    return int(report["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
