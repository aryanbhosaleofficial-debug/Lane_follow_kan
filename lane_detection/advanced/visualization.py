"""Optional diagnostic drawing; always copies shared M02 images."""
from __future__ import annotations

import cv2
import numpy as np

from .models import DebugImages, LaneObservation, Polynomial


def _curve(image: np.ndarray, poly: Polynomial, color: tuple[int, int, int],
           intervals: tuple[tuple[float, float], ...], dashed: bool = False) -> None:
    for start, end in intervals:
        start, end = max(start, poly.v_min), min(end, poly.v_max)
        if end <= start:
            continue
        # Include the exact endpoint without stepping past a limited domain.
        rows = np.linspace(start, end, max(2, int(np.ceil((end - start) / 2)) + 1))
        points = np.rint(np.column_stack((poly.evaluate(rows), rows))).astype(np.int32)
        if dashed:
            for i in range(0, len(points) - 1, 6):
                cv2.polylines(image, [points[i:i + 3]], False, color, 2, cv2.LINE_AA)
        elif len(points) > 1:
            cv2.polylines(image, [points], False, color, 2, cv2.LINE_AA)


def draw_observation(image: np.ndarray, observation: LaneObservation, debug: DebugImages | None = None) -> np.ndarray:
    """Green observed curves, orange dashed gaps/inference, cyan centreline."""
    output = image.copy()
    if debug is not None:
        output[debug.rejected_mask != 0] = (80, 30, 130)
        output[debug.accepted_mask != 0] = (100, 180, 100)
    for b in observation.boundaries:
        _curve(output, b.polynomial, (0, 255, 0) if b.marking_evidence_valid else (0, 0, 255), b.observed_intervals)
        _curve(output, b.polynomial, (0, 160, 255), b.unsupported_intervals, dashed=True)
        row = b.polynomial.v_max
        u = int(round(float(b.polynomial.evaluate(row))))
        role = "candidate"
        pair = observation.selected_pair
        if pair is not None and b.candidate_id in {pair.left_id, pair.right_id}:
            role = "left" if b.candidate_id == pair.left_id else "right"
        cv2.putText(output, f"{b.candidate_id}:{role}", (max(0, u - 20), int(row) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
    if observation.selected_pair is not None:
        p = observation.selected_pair.centreline
        _curve(output, p, (255, 255, 0), ((p.v_min, p.v_max),), observation.selected_pair.inferred)
    reasons = ",".join(observation.reason_codes)
    max_characters = max(16, int(image.shape[1] / 6))
    if len(reasons) > max_characters:
        reasons = reasons[:max_characters - 3] + "..."
    status = observation.validity.upper() + (" LIMITED" if observation.limited_support else "")
    lines = (f"{status} score={observation.confidence:.2f} {observation.timings.total_ms:.1f}ms",
             f"{observation.calibration_status} | pixels",
             reasons)
    for i, text in enumerate(lines):
        cv2.putText(output, text, (5, 15 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, .36, (0, 0, 0), 3)
        cv2.putText(output, text, (5, 15 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255), 1)
    return output
