"""Command-line tools for M02. Run from the repository root:  python -m camera_calibration <command> --help

Commands
--------
intrinsic        chessboard images -> intrinsic calibration -> profile (+ corner / reprojection overlays)
road-plane       measured image<->ground points -> road-plane homography (+ optional BEV grid) -> profile
set-bev          change the BEV grid of a profile without recalibrating
info             print a profile's contents and run the self-consistency checks
bev              warp one image into the BEV and write diagnostic overlays
check-distances  compare tape-measured distances with the calibration
live             run the real-time pipeline on a camera index / video file through M01
pick-points      click image points and write a points-file TEMPLATE (ground positions you must measure and fill in)
synthetic-demo   run the whole workflow on SIMULATED data (proves the software; calibrates nothing real)

Exit status: 0 success, 2 for calibration/validation problems (message on stderr), 130 on Ctrl-C.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import diagnostics as diag
from .bev import BEVConfig
from .errors import CalibrationError, CalibrationQualityError, FrameUnavailableError, InvalidCalibrationInputError
from .intrinsics import IntrinsicCalibrationConfig, IntrinsicCalibrationResult, IntrinsicCalibrator, load_gray_image
from .mapper import BEVMapper
from .plane_calibration import METHODS, RoadPlaneCalibrator, RoadPlaneConfig
from .profile import CalibrationProfile, new_profile
from .roadplane import CAMERA_FACING
from .undistort import INTERPOLATIONS
from .validation import CalibrationValidator, DistanceMeasurement


# -- argument helpers -----------------------------------------------------------------------------
def _pattern(text: str) -> tuple[int, int]:
    try:
        cols, rows = (int(v) for v in text.lower().split("x"))
        return cols, rows
    except ValueError:
        raise argparse.ArgumentTypeError("pattern must look like 9x6 (INNER corners: columns x rows)") from None


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().split("x"))
        return w, h
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 1280x720") from None


def _read_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"{c} not allowed")))
    except OSError as exc:
        raise InvalidCalibrationInputError(f"cannot read {path}: {exc}") from None
    except ValueError as exc:
        raise InvalidCalibrationInputError(f"{path} is not valid JSON: {exc}") from None


def _pair(value: Any, where: str) -> list[float]:
    if value is None:
        raise InvalidCalibrationInputError(f"{where} is null: fill in the measured value before calibrating")
    if (not isinstance(value, list) or len(value) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)):
        raise InvalidCalibrationInputError(f"{where} must be two finite numbers, got {value!r}")
    return [float(v) for v in value]


def load_correspondences(path: str, key: str = "correspondences") -> dict[str, Any]:
    """Parse and validate a points file (see README). Returns image/ground arrays plus metadata."""
    data = _read_json(path)
    if not isinstance(data, dict):
        raise InvalidCalibrationInputError("points file must be a JSON object")
    space = data.get("space", "raw")
    if space not in ("raw", "undistorted"):
        raise InvalidCalibrationInputError("space must be 'raw' or 'undistorted'")

    def parse(entries: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(entries, list):
            raise InvalidCalibrationInputError(f"{name} must be a list")
        img, gnd = [], []
        for i, e in enumerate(entries):
            if not isinstance(e, dict) or "image_px" not in e or "ground_m" not in e:
                raise InvalidCalibrationInputError(f"{name}[{i}] needs 'image_px' and 'ground_m'")
            img.append(_pair(e["image_px"], f"{name}[{i}].image_px"))
            gnd.append(_pair(e["ground_m"], f"{name}[{i}].ground_m"))
        return np.array(img, float).reshape(-1, 2), np.array(gnd, float).reshape(-1, 2)

    img, gnd = parse(data.get(key, []), key)
    held = parse(data["held_out"], "held_out") if "held_out" in data else None
    mounting = data.get("mounting", {})
    if not isinstance(mounting, dict):
        raise InvalidCalibrationInputError("mounting must be an object")
    origin = data.get("origin_description", "")
    if not isinstance(origin, str):
        raise InvalidCalibrationInputError("origin_description must be a string")
    return {"image": img, "ground": gnd, "held_out": held, "space": space, "mounting": mounting, "origin_description": origin}


def _bev_from_args(args: argparse.Namespace) -> BEVConfig | None:
    if args.bev_extents is None and args.mpp is None:
        return None
    if args.bev_extents is None or args.mpp is None:
        raise InvalidCalibrationInputError("give both --bev-extents XMIN XMAX YMIN YMAX and --mpp")
    xmin, xmax, ymin, ymax = args.bev_extents
    return BEVConfig.from_extents(xmin, xmax, ymin, ymax, args.mpp, interpolation=args.interpolation)


def _add_bev_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bev-extents", nargs=4, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   help="BEV coverage in metres: forward range, then lateral range (Y positive = left)")
    p.add_argument("--mpp", type=float, help="BEV scale in metres per pixel")
    p.add_argument("--interpolation", choices=sorted(INTERPOLATIONS), default="linear")


# -- commands -------------------------------------------------------------------------------------
def cmd_intrinsic(args: argparse.Namespace) -> int:
    cfg = IntrinsicCalibrationConfig(args.pattern, args.square, args.unit, model=args.model, min_views=args.min_views, detector=args.detector,
                                     expected_image_size=args.expected_size, max_rms_px=args.max_rms, enforce_quality=not args.no_enforce_quality)
    calibrator = IntrinsicCalibrator(cfg)
    try:
        result = calibrator.calibrate_directory(args.images)
    except CalibrationQualityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.result is not None:
            print(exc.result.report(), file=sys.stderr)
            _write_overlays(exc.result, Path(args.images), args.overlay_dir)
        return 2
    print(result.report()); print(); print(diag.reprojection_summary(result))
    _write_overlays(result, Path(args.images), args.overlay_dir)
    profile = new_profile(result.intrinsics, args.camera_id, alpha=args.alpha, source_ids=tuple(args.source_id or ()))
    path = profile.save(args.out)
    print(f"\nwrote {path}  (alpha={args.alpha}; choose it BEFORE the road-plane step: changing it later invalidates the homography)")
    if result.quality.status != "ok":
        print(f"note: quality status is {result.quality.status.upper()}; read the issues above before relying on this profile")
    return 0


def _projected(result: IntrinsicCalibrationResult, index: int) -> np.ndarray:
    from .intrinsics import chessboard_object_points
    i = result.intrinsics
    obj = chessboard_object_points(i.pattern_size, i.square_size_m)
    rv, tv = result.rvecs[index].reshape(3, 1), result.tvecs[index].reshape(3, 1)
    if i.is_fisheye:
        pts, _ = cv2.fisheye.projectPoints(obj.reshape(1, -1, 3), rv, tv, i.camera_matrix, i.dist_coeffs)
    else:
        pts, _ = cv2.projectPoints(obj, rv, tv, i.camera_matrix, i.dist_coeffs)
    return pts.reshape(-1, 2)


def _write_overlays(result: IntrinsicCalibrationResult, folder: Path, overlay_dir: str | None) -> None:
    if not overlay_dir:
        return
    out = Path(overlay_dir)
    used = [v for v in result.views if v.used]
    for k, (view, corners) in enumerate(zip(used, result.corners)):
        gray = load_gray_image(folder / view.name)
        if gray is None:
            continue
        pattern = result.intrinsics.pattern_size
        diag.save_image(out / f"{Path(view.name).stem}_corners.png", diag.draw_corner_overlay(gray, corners, pattern))
        diag.save_image(out / f"{Path(view.name).stem}_reprojection.png", diag.draw_reprojection_error(gray, corners, _projected(result, k)))
    print(f"wrote {len(used)} corner / reprojection overlay pair(s) to {out}")


def cmd_road_plane(args: argparse.Namespace) -> int:
    profile = CalibrationProfile.load(args.profile)
    pts = load_correspondences(args.points)
    cfg = RoadPlaneConfig(method=args.method, camera_facing=args.facing, origin_description=pts["origin_description"] or args.origin or "",
                          max_condition_number=args.max_condition)
    calibrator = RoadPlaneCalibrator(profile.intrinsics, profile.undistortion, cfg)
    fit = calibrator.estimate(pts["image"], pts["ground"], points_space=pts["space"])
    print(fit.summary())
    mounting = {**pts["mounting"], **({"camera_height_m": args.camera_height} if args.camera_height else {})}
    height = mounting.get("camera_height_m")
    if isinstance(height, (int, float)) and height > 0:
        est = fit.metrics["camera_height_m"]
        flag = "OK" if abs(est - height) <= 0.15 * height else "MISMATCH - check intrinsics, points and measurements"
        print(f"\ncamera height: homography implies {est:.3f} m, you measured {height:.3f} m  [{flag}]")
    if pts["held_out"] is not None:
        report = calibrator.validate(fit, *pts["held_out"], points_space=pts["space"])
        print(f"\nheld-out validation over {len(report['errors_m'])} point(s): RMS {report['rms_m'] * 1000:.1f} mm, max {report['max_m'] * 1000:.1f} mm "
              f"(points outside the calibrated hull: {report['inside_calibrated_hull'].count(False)})")
    updated = profile.with_road_plane(fit)
    if mounting:
        updated = updated.with_mounting(**mounting)
    bev = _bev_from_args(args)
    if bev is not None:
        updated = updated.with_bev(bev)
        mapper = BEVMapper(updated)
        print(f"\nBEV {bev.width_px}x{bev.height_px} px at {bev.meters_per_pixel * 1000:.2f} mm/px: observed {mapper.info.valid_fraction:.0%}, "
              f"supported {mapper.info.support_fraction:.0%}")
        for w in mapper.info.warnings:
            print(f"  warning: {w}")
    else:
        print("\nno BEV grid yet: add --bev-extents/--mpp here or run `set-bev`")
    path = updated.save(args.out)
    print(f"\nwrote {path}")
    if args.image:
        img = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise InvalidCalibrationInputError(f"cannot read image {args.image}")
        diag.save_image(Path(args.overlay_dir) / "reference_points.png", diag.draw_reference_points(img, updated))
        diag.save_image(Path(args.overlay_dir) / "ground_grid.png", diag.draw_ground_grid_on_image(img, updated))
        print(f"wrote overlays to {args.overlay_dir}")
    return 0


def cmd_set_bev(args: argparse.Namespace) -> int:
    profile = CalibrationProfile.load(args.profile)
    bev = _bev_from_args(args)
    if bev is None:
        raise InvalidCalibrationInputError("--bev-extents and --mpp are required")
    updated = profile.with_bev(bev)
    mapper = BEVMapper(updated)
    print(f"BEV {bev.width_px}x{bev.height_px} px; observed {mapper.info.valid_fraction:.0%}, supported {mapper.info.support_fraction:.0%}")
    for w in mapper.info.warnings:
        print(f"  warning: {w}")
    print(f"wrote {updated.save(args.out)}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    p = CalibrationProfile.load(args.profile)
    i = p.intrinsics
    print(f"profile {p.camera_id!r}  fingerprint {p.fingerprint()}  created {p.created_utc}  size {i.image_width}x{i.image_height}")
    print(f"  bound sources: {list(p.source_ids) or 'none (any source accepted)'}")
    print(f"  intrinsics: model={i.model} fx={i.camera_matrix[0, 0]:.2f} fy={i.camera_matrix[1, 1]:.2f} cx={i.camera_matrix[0, 2]:.2f} "
          f"cy={i.camera_matrix[1, 2]:.2f} rms={i.rms_reprojection_px} views={i.num_views} quality={i.quality.status if i.quality else 'n/a'}")
    print(f"  undistortion: alpha={p.undistortion.alpha}")
    if p.mounting:
        print(f"  mounting notes: {p.mounting}")
    print(p.road_plane.summary() if p.road_plane else "  road plane: NOT CALIBRATED")
    if p.bev:
        print(f"  BEV {p.bev.width_px}x{p.bev.height_px} px, {p.bev.meters_per_pixel * 1000:.2f} mm/px, x {p.bev.x_min_m:.2f}..{p.bev.x_max_m:.2f} m, "
              f"y {p.bev.y_min_m:.2f}..{p.bev.y_max_m:.2f} m")
    if p.is_metric:
        print(CalibrationValidator(p).check_consistency())
        return 0
    print("  (profile is not metric yet: self-consistency checks skipped)")
    return 0


def _read_color(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidCalibrationInputError(f"cannot read image {path}")
    return img


def cmd_bev(args: argparse.Namespace) -> int:
    profile = CalibrationProfile.load(args.profile)
    raw = _read_color(args.image)
    mapper = BEVMapper(profile)
    bev = mapper.warp(raw)
    out = Path(args.out)
    diag.save_image(out, bev)
    stem = out.with_suffix("")
    diag.save_image(Path(f"{stem}_grid.png"), diag.draw_bev_grid(bev, profile.bev, spacing_m=args.grid))
    diag.save_image(Path(f"{stem}_regions.png"), diag.draw_valid_region(bev_image=bev, valid_mask=mapper.valid_mask, support_mask=mapper.support_mask))
    diag.save_image(Path(f"{stem}_sidebyside.png"), diag.side_by_side(raw, bev))
    print(f"BEV {bev.shape[1]}x{bev.shape[0]}: observed {mapper.info.valid_fraction:.0%}, supported {mapper.info.support_fraction:.0%}; "
          f"source footprint per BEV pixel median {mapper.info.footprint_px_median:.2f} px")
    for w in mapper.info.warnings:
        print(f"  warning: {w}")
    print(f"wrote {out} and *_grid / *_regions / *_sidebyside next to it")
    return 0


def cmd_check_distances(args: argparse.Namespace) -> int:
    profile = CalibrationProfile.load(args.profile)
    data = _read_json(args.measurements)
    entries = data.get("distances") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise InvalidCalibrationInputError("measurements file needs a non-empty 'distances' list")
    items = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict) or not {"a_px", "b_px", "distance_m"} <= set(e):
            raise InvalidCalibrationInputError(f"distances[{i}] needs a_px, b_px and distance_m")
        items.append(DistanceMeasurement(tuple(_pair(e["a_px"], f"distances[{i}].a_px")), tuple(_pair(e["b_px"], f"distances[{i}].b_px")),
                                         e["distance_m"], str(e.get("label", "")), bool(e.get("undistorted", False))))
    report = CalibrationValidator(profile).check_distances(items, args.tolerance_m)
    print(report)
    return 0 if report.passed else 2


def cmd_live(args: argparse.Namespace) -> int:
    try:
        from camera_input import CameraConfig, CameraManager
    except ImportError:
        raise InvalidCalibrationInputError("the live command needs the M01 package (camera_input) importable from the repository root") from None
    from .pipeline import BEVProcessor, BEVWorker, ProcessorConfig
    profile = CalibrationProfile.load(args.profile)
    source = int(args.source) if args.source.isdecimal() and not args.file else args.source
    cfg = CameraConfig(source=source, kind="file" if args.file else "live", width=args.size[0] if args.size else None,
                       height=args.size[1] if args.size else None, pace_video=not args.fast)
    processor = BEVProcessor(profile, ProcessorConfig(max_frame_age_s=None if args.file else args.max_age))
    show = not args.no_gui and args.frames is None
    last = 0
    final = None
    with CameraManager(cfg) as camera, BEVWorker(processor, camera) as worker:
        t0, shown = time.monotonic(), 0
        try:
            while args.frames is None or shown < args.frames:
                try:
                    bev = worker.wait_for_bev(last, timeout=2.0)
                except EOFError:
                    print("end of video"); break
                last = bev.bev_sequence
                final = bev
                shown += 1
                raw = camera.latest()
                if show:
                    panel = diag.side_by_side(raw.image if raw is not None else bev.image, diag.draw_valid_region(bev), height=480)
                    cv2.imshow("M02 raw | BEV (magenta=unobserved, green=extrapolated)", panel)
                    if cv2.waitKey(1) & 255 in (ord("q"), 27):
                        break
        finally:
            if show:
                cv2.destroyAllWindows()
        if args.save_dir and final is not None:
            # M01 keeps only the newest frame, so an unpaced source can end before N BEVs were produced: save whatever came last
            diag.save_image(Path(args.save_dir) / "last_bev.png", final.image)
        elapsed = time.monotonic() - t0
        if shown == 0:
            raise FrameUnavailableError("no BEV frame was produced before the source ended or failed", reason="no_frame")
        s = processor.stats()
        print(f"{shown} BEV frame(s) in {elapsed:.2f} s; processing mean {s['mean_processing_ms']:.2f} ms max {s['max_processing_ms']:.2f} ms; "
              f"errors {s['errors_by_type'] or 'none'}; session changes {s['session_changes']}")
    return 0


def cmd_pick_points(args: argparse.Namespace) -> int:
    img = _read_color(args.image)
    points: list[tuple[float, float]] = []
    name = "click road points; u=undo, w/Enter=save, q=quit"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((float(x), float(y)))

    cv2.namedWindow(name)
    cv2.setMouseCallback(name, on_mouse)
    while True:
        view = img.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(view, (int(x), int(y)), 5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(view, str(i), (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(name, view)
        key = cv2.waitKey(30) & 255
        if key == ord("u") and points:
            points.pop()
        elif key in (ord("w"), 13):
            break
        elif key in (ord("q"), 27):
            cv2.destroyAllWindows()
            print("aborted; nothing written")
            return 1
    cv2.destroyAllWindows()
    Path(args.out).write_text(json.dumps(points_template(points, args.space), indent=2), encoding="utf-8")
    print(f"wrote {len(points)} point(s) to {args.out}: now measure each point's ground position and replace the null ground_m values")
    return 0


def points_template(points: list[tuple[float, float]], space: str = "raw") -> dict[str, Any]:
    """A points file with the picked pixels and NULL ground positions: calibration refuses it until you measure and fill them in."""
    return {"space": space, "origin_description": "DESCRIBE where (0, 0) is, e.g. 'floor point under the camera lens'",
            "correspondences": [{"image_px": [x, y], "ground_m": None} for x, y in points],
            "held_out": [], "mounting": {"camera_height_m": None}}


def cmd_synthetic_demo(args: argparse.Namespace) -> int:
    from .demo import run_synthetic_demo
    print("SYNTHETIC DEMONSTRATION: simulated camera and road; calibrates nothing real.\n")
    summary = run_synthetic_demo(args.out, seed=args.seed)
    print(Path(summary["report_path"]).read_text(encoding="utf-8"))
    print(f"artefacts written to {summary['out_dir']}")
    return 0 if summary["consistency_passed"] and summary["distances_passed"] else 2


# -- entry point -----------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m camera_calibration", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("intrinsic", help="chessboard images -> intrinsic profile")
    p.add_argument("--images", required=True, help="folder of chessboard images (non-recursive)")
    p.add_argument("--pattern", type=_pattern, required=True, help="INNER corners as COLSxROWS, e.g. 9x6")
    p.add_argument("--square", type=float, required=True, help="printed square size (a number)")
    p.add_argument("--unit", choices=("m", "cm", "mm"), required=True, help="unit of --square (mandatory, no default)")
    p.add_argument("--model", choices=("radial2", "standard5", "rational8", "fisheye"), default="standard5")
    p.add_argument("--camera-id", required=True, help="a name for this camera (stored in the profile; no URLs with credentials)")
    p.add_argument("--out", required=True, help="profile JSON to write")
    p.add_argument("--alpha", type=float, default=0.0, help="undistortion crop: 0 = no black border, 1 = keep all pixels")
    p.add_argument("--min-views", type=int, default=10)
    p.add_argument("--detector", choices=("classic", "sb"), default="classic")
    p.add_argument("--expected-size", type=_size, help="WxH the camera will run at; other image sizes are rejected")
    p.add_argument("--max-rms", type=float, default=1.0, help="screening limit for overall RMS reprojection error (px)")
    p.add_argument("--no-enforce-quality", action="store_true", help="write the profile even if quality checks fail (still reported)")
    p.add_argument("--overlay-dir", help="write corner and reprojection overlays here")
    p.add_argument("--source-id", action="append", help="bind the profile to an M01 source id (repeatable)")
    p.set_defaults(func=cmd_intrinsic)

    p = sub.add_parser("road-plane", help="measured points -> road-plane homography")
    p.add_argument("--profile", required=True, help="intrinsic profile from the `intrinsic` step")
    p.add_argument("--points", required=True, help="points JSON (see README)")
    p.add_argument("--out", required=True)
    p.add_argument("--method", choices=METHODS, default="least_squares")
    p.add_argument("--facing", choices=CAMERA_FACING, default="forward")
    p.add_argument("--max-condition", type=float, default=1e5)
    p.add_argument("--origin", help="description of the ground origin if not in the points file")
    p.add_argument("--camera-height", type=float, help="measured lens height above the road in metres (cross-checked against the fit)")
    p.add_argument("--image", help="a raw image of the road to draw reference-point / grid overlays on")
    p.add_argument("--overlay-dir", default="overlays")
    _add_bev_args(p)
    p.set_defaults(func=cmd_road_plane)

    p = sub.add_parser("set-bev", help="change the BEV grid of a profile")
    p.add_argument("--profile", required=True); p.add_argument("--out", required=True)
    _add_bev_args(p)
    p.set_defaults(func=cmd_set_bev)

    p = sub.add_parser("info", help="describe a profile and run self-consistency checks")
    p.add_argument("--profile", required=True)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("bev", help="warp one raw image into the BEV")
    p.add_argument("--profile", required=True); p.add_argument("--image", required=True); p.add_argument("--out", required=True)
    p.add_argument("--grid", type=float, default=0.25, help="BEV grid spacing in metres")
    p.set_defaults(func=cmd_bev)

    p = sub.add_parser("check-distances", help="compare tape-measured distances with the calibration")
    p.add_argument("--profile", required=True); p.add_argument("--measurements", required=True)
    p.add_argument("--tolerance-m", type=float, required=True, help="your acceptance tolerance in metres (no default: it is your decision)")
    p.set_defaults(func=cmd_check_distances)

    p = sub.add_parser("live", help="real-time raw | BEV through M01")
    p.add_argument("--profile", required=True)
    p.add_argument("--source", required=True, help="camera index, stream URL, or video path (with --file)")
    p.add_argument("--file", action="store_true"); p.add_argument("--fast", action="store_true", help="do not pace video files")
    p.add_argument("--size", type=_size); p.add_argument("--max-age", type=float, default=0.5)
    p.add_argument("--frames", type=int, help="headless: process this many frames and exit")
    p.add_argument("--no-gui", action="store_true"); p.add_argument("--save-dir")
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("pick-points", help="click points on an image -> points-file template")
    p.add_argument("--image", required=True); p.add_argument("--out", required=True)
    p.add_argument("--space", choices=("raw", "undistorted"), default="raw")
    p.set_defaults(func=cmd_pick_points)

    p = sub.add_parser("synthetic-demo", help="run the full workflow on SIMULATED data")
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synthetic_demo)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CalibrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
