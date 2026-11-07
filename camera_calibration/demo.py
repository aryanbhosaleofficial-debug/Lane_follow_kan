"""End-to-end SYNTHETIC demonstration of the M02 workflow.

Everything here is simulated: a virtual camera with a made-up lens, a virtual road with made-up
dimensions, and "measurements" computed from that virtual scene. The output proves the software
chain works and shows what the tools produce; it says NOTHING about any real camera, mounting
pose or track, and must never be used as a calibration for one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import diagnostics as diag
from .bev import BEVConfig
from .intrinsics import IntrinsicCalibrationConfig, IntrinsicCalibrator
from .mapper import BEVMapper
from .plane_calibration import RoadPlaneCalibrator, RoadPlaneConfig
from .profile import new_profile
from .synthetic import (SYNTHETIC_NOTICE, SyntheticCamera, SyntheticPose, make_road_scene, project_ground_points,
                        random_board_poses, render_chessboard)
from .validation import CalibrationValidator, DistanceMeasurement

PATTERN, SQUARE_MM = (9, 6), 25.0
CAMERA = SyntheticCamera(dist=(-0.28, 0.09, 0.001, -0.0008, -0.01))
POSE = SyntheticPose(height_m=0.25, pitch_down_deg=28.0)
LANE_WIDTH_M = 0.50              # SYNTHETIC lane width of the virtual road


def run_synthetic_demo(out_dir: str | Path, *, seed: int = 0, noise_px: float = 0.3, views: int = 16) -> dict[str, Any]:
    """Run the whole workflow on simulated data and write every artefact under ``out_dir``. Returns a summary dict."""
    out = Path(out_dir)
    img_dir = out / "intrinsic_images"
    overlay_dir = out / "overlays"
    img_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    (out / "SYNTHETIC_README.txt").write_text(SYNTHETIC_NOTICE + "\nNothing in this folder describes a real camera or track.\n", encoding="utf-8")

    # 1. chessboard images -> intrinsic calibration
    for i, (R, t) in enumerate(random_board_poses(CAMERA, PATTERN, SQUARE_MM / 1000.0, views, seed=seed)):
        diag.save_image(img_dir / f"board_{i:02d}.png", render_chessboard(CAMERA, PATTERN, SQUARE_MM / 1000.0, R, t, seed=seed + i))
    calibrator = IntrinsicCalibrator(IntrinsicCalibrationConfig(PATTERN, SQUARE_MM, "mm"))
    intrinsic = calibrator.calibrate_directory(img_dir)
    profile = new_profile(intrinsic.intrinsics, "SYNTHETIC-camera", note=SYNTHETIC_NOTICE)

    # 2. road plane from simulated "measurements"
    gx, gy = np.meshgrid(np.linspace(0.5, 1.6, 4), np.linspace(-0.35, 0.35, 3))
    ground = np.column_stack([gx.ravel(), gy.ravel()])
    rng = np.random.default_rng(seed)
    picked = project_ground_points(CAMERA, POSE, ground) + rng.normal(0, noise_px, (len(ground), 2))
    (out / "points.json").write_text(json.dumps({
        "_comment": SYNTHETIC_NOTICE, "space": "raw", "origin_description": "SYNTHETIC: ground point below the virtual camera",
        "correspondences": [{"image_px": p.tolist(), "ground_m": g.tolist()} for p, g in zip(picked, ground)]}, indent=2), encoding="utf-8")
    plane = RoadPlaneCalibrator(profile.intrinsics, profile.undistortion, RoadPlaneConfig(origin_description="SYNTHETIC origin")).estimate(picked, ground)
    profile = profile.with_road_plane(plane).with_bev(BEVConfig.from_extents(0.3, 2.3, -0.7, 0.7, 0.004))
    profile.save(out / "profile.json")

    # 3. BEV of a simulated road, with diagnostics
    straight = make_road_scene("straight").render(CAMERA, POSE)
    curve = make_road_scene("curve").render(CAMERA, POSE)
    mapper = BEVMapper(profile)
    for name, raw in (("straight", straight), ("curve", curve)):
        bev = mapper.warp(raw)
        diag.save_image(overlay_dir / f"{name}_raw_with_grid.png", diag.draw_ground_grid_on_image(raw, profile))
        diag.save_image(overlay_dir / f"{name}_bev_grid.png", diag.draw_bev_grid(bev, profile.bev))
        diag.save_image(overlay_dir / f"{name}_bev_regions.png", diag.draw_valid_region(bev_image=bev, valid_mask=mapper.valid_mask, support_mask=mapper.support_mask))
        diag.save_image(overlay_dir / f"{name}_side_by_side.png", diag.side_by_side(raw, bev))
    diag.save_image(overlay_dir / "reference_points.png", diag.draw_reference_points(straight, profile))

    # 4. validation
    validator = CalibrationValidator(profile)
    consistency = validator.check_consistency()
    a, b = project_ground_points(CAMERA, POSE, [[1.0, -LANE_WIDTH_M / 2], [1.0, LANE_WIDTH_M / 2]])
    distances = validator.check_distances([DistanceMeasurement(tuple(a), tuple(b), LANE_WIDTH_M, "SIMULATED lane width")], tolerance_m=0.005)

    report = "\n".join([
        SYNTHETIC_NOTICE, "", diag.reprojection_summary(intrinsic), "", intrinsic.report(), "", plane.summary(), "",
        f"BEV: {profile.bev.shape[1]}x{profile.bev.shape[0]} px at {profile.bev.meters_per_pixel * 1000:.1f} mm/px; observed {mapper.info.valid_fraction:.0%}, "
        f"supported {mapper.info.support_fraction:.0%}; map build {mapper.info.build_seconds * 1000:.0f} ms",
        *[f"  BEV note: {w}" for w in mapper.info.warnings], "", str(consistency), "", str(distances), "",
        "Truth used by the simulation (for comparison only): camera height 0.25 m, pitch 28 deg, fx=fy=520 px, k1=-0.28.", ""])
    (out / "report.txt").write_text(report, encoding="utf-8")
    return {"profile_path": str(out / "profile.json"), "intrinsic_status": intrinsic.quality.status, "intrinsic_rms_px": intrinsic.intrinsics.rms_reprojection_px,
            "road_plane_metrics": dict(plane.metrics), "consistency_passed": consistency.passed, "distances_passed": distances.passed,
            "bev_valid_fraction": mapper.info.valid_fraction, "report_path": str(out / "report.txt"), "out_dir": str(out)}
