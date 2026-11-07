"""Video appearance-change diagnostic; does not infer camera-to-vehicle mounting stability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .video_validation import inspect_video


def diagnose_video(video: Path, frame_ids: list[int], *, stationary_points: Path | None = None,
                   max_shift_px: float = 3.0) -> dict:
    info = inspect_video(video)
    if not frame_ids or len(set(frame_ids)) != len(frame_ids) or any(n < 1 or n > info["frames_reported"] for n in frame_ids):
        raise ValueError("frame IDs must be distinct and within the reported video frame range")
    cap = cv2.VideoCapture(str(video))
    orb = cv2.ORB_create(nfeatures=1000)
    selected = []
    try:
        for n in sorted(frame_ids):
            cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"could not decode frame {n}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, descriptors = orb.detectAndCompute(gray, None)
            selected.append((n, kp, descriptors))
    finally:
        cap.release()
    pairs = []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    for (a_id, a_kp, a_desc), (b_id, b_kp, b_desc) in zip(selected, selected[1:]):
        good = []
        if a_desc is not None and b_desc is not None and len(a_desc) >= 2 and len(b_desc) >= 2:
            for pair in matcher.knnMatch(a_desc, b_desc, k=2):
                if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance:
                    good.append(pair[0])
        row = {"from_frame": a_id, "to_frame": b_id, "feature_matches": len(good),
               "median_displacement_px": None, "homography_inlier_fraction": None}
        if len(good) >= 8:
            before = np.float32([a_kp[m.queryIdx].pt for m in good])
            after = np.float32([b_kp[m.trainIdx].pt for m in good])
            row["median_displacement_px"] = float(np.median(np.linalg.norm(after - before, axis=1)))
            _, mask = cv2.findHomography(before, after, cv2.RANSAC, 3.0)
            if mask is not None:
                row["homography_inlier_fraction"] = float(mask.mean())
        pairs.append(row)
    report = {"video": info, "frames_analyzed": sorted(frame_ids), "comparisons": pairs,
            "conclusion": "INCONCLUSIVE_FOR_MOUNTING_STABILITY",
            "uncertainty": ("Feature matches track scene appearance. A moving vehicle naturally changes them; "
                            "they cannot separate vehicle motion from camera-to-vehicle mounting changes. "
                            "Stationary target footage or measured vehicle/ground references are required."),
            "action": "Inspect mounting and recalibrate with the camera fixed in its final vehicle pose."}
    if stationary_points is not None:
        if not np.isfinite(max_shift_px) or max_shift_px <= 0:
            raise ValueError("max_shift_px must be finite and positive")
        data = json.loads(stationary_points.read_text(encoding="utf-8"))
        if data.get("target_stationary") is not True:
            raise ValueError("reference file must explicitly assert target_stationary=true")
        entries = data.get("frames")
        if not isinstance(entries, list) or sorted(e.get("frame_id") for e in entries) != sorted(frame_ids):
            raise ValueError("reference file must contain exactly the analyzed frame IDs")
        points = []
        for entry in sorted(entries, key=lambda e: e["frame_id"]):
            p = np.asarray(entry.get("points_px"), dtype=np.float64)
            if p.ndim != 2 or p.shape[1] != 2 or len(p) < 4 or not np.isfinite(p).all():
                raise ValueError("each frame needs >=4 finite corresponding target points")
            points.append(p)
        if len({p.shape for p in points}) != 1:
            raise ValueError("reference point count must match between frames")
        shifts = [float(np.max(np.linalg.norm(p - points[0], axis=1))) for p in points]
        report["stationary_target"] = {"max_point_shift_px_by_frame": dict(zip(sorted(frame_ids), shifts)),
                                       "threshold_px": max_shift_px,
                                       "assumption": "operator states target and camera platform were stationary"}
        report["conclusion"] = ("STABLE_RELATIVE_TO_STATIONARY_TARGET" if max(shifts) <= max_shift_px
                                else "VIEWPOINT_OR_TARGET_CHANGED")
        report["uncertainty"] = ("This checks only supplied target correspondences. It cannot distinguish a moved target "
                                 "from a moved camera, or certify the camera mount during later vehicle motion.")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[1, 31, 61, 121, 241, 361, 481, 590])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stationary-points", type=Path, help="JSON with stationary target correspondences for every chosen frame")
    ap.add_argument("--max-shift-px", type=float, default=3.0)
    args = ap.parse_args(argv)
    report = diagnose_video(args.video, args.frames, stationary_points=args.stationary_points,
                            max_shift_px=args.max_shift_px)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {report['conclusion']}")


if __name__ == "__main__": main()
