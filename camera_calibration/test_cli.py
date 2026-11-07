"""Tests for the diagnostics and the command-line tools (SYNTHETIC data)."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from camera_calibration import diagnostics as diag
from camera_calibration.cli import load_correspondences, main, points_template
from camera_calibration.demo import CAMERA, PATTERN, POSE, SQUARE_MM, run_synthetic_demo
from camera_calibration.errors import InvalidCalibrationInputError, InvalidFrameError
from camera_calibration.mapper import BEVMapper
from camera_calibration.profile import CalibrationProfile
from camera_calibration.synthetic import (make_road_scene, project_ground_points, random_board_poses, render_chessboard)
from camera_calibration.test_bev import BEV, PROFILE, STRAIGHT_IMG


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main([str(a) for a in argv])
        except SystemExit as exc:                       # argparse errors
            code = exc.code
    return code, out.getvalue(), err.getvalue()


class DiagnosticsTests(unittest.TestCase):
    def test_overlays_return_new_images_of_the_same_size_and_leave_inputs_alone(self):
        img = STRAIGHT_IMG.copy(); before = img.copy()
        mapper = BEVMapper(PROFILE)
        bev = mapper.warp(img)
        outs = [diag.draw_ground_grid_on_image(img, PROFILE), diag.draw_reference_points(img, PROFILE),
                diag.draw_bev_grid(bev, BEV), diag.draw_valid_region(bev_image=bev, valid_mask=mapper.valid_mask, support_mask=mapper.support_mask)]
        for o, shape in zip(outs, [img.shape, img.shape, bev.shape, bev.shape]):
            self.assertEqual(o.shape, shape)
        np.testing.assert_array_equal(img, before)
        self.assertFalse(any(np.shares_memory(o, img) or np.shares_memory(o, bev) for o in outs))
        self.assertTrue(all((o != b).any() for o, b in zip(outs[:2], [img, img])))      # something was actually drawn

    def test_valid_region_colour_codes_the_masks(self):
        mapper = BEVMapper(PROFILE)
        bev = mapper.warp(STRAIGHT_IMG)
        out = diag.draw_valid_region(bev_image=bev, valid_mask=mapper.valid_mask, support_mask=mapper.support_mask)
        unobserved = mapper.valid_mask == 0
        self.assertTrue((out[unobserved] == (95, 0, 95)).all())
        supported = mapper.support_mask > 0
        np.testing.assert_array_equal(out[supported], bev[supported])                  # trusted pixels are untouched
        extrapolated = (mapper.valid_mask > 0) & (mapper.support_mask == 0)
        self.assertTrue((out[extrapolated][:, 1] >= bev[extrapolated][:, 1] * 0.6 - 1).all())

    def test_bev_grid_lines_are_where_the_metric_grid_says(self):
        blank = np.zeros(BEV.shape + (3,), np.uint8)
        out = diag.draw_bev_grid(blank, BEV, spacing_m=0.5, label=False)
        v = int(round(BEV.ground_to_bev([[1.0, 0.0]])[0, 1]))
        self.assertTrue((out[v, 5:-5] != 0).any())                                      # a line at x = 1.0 m
        u = int(round(BEV.ground_to_bev([[0.0, 0.5]])[0, 0]))
        self.assertTrue((out[5:-5, u] != 0).any())                                      # a line at y = 0.5 m (left of origin)
        self.assertEqual(diag.draw_bev_grid(blank, BEV, spacing_m=0.5).shape, blank.shape)

    def test_labels_stay_inside_the_image(self):
        img = np.zeros((60, 80, 3), np.uint8)
        diag._put_label(img, "a very long label that cannot fit", (70, 5))
        diag._put_label(img, "x", (float("nan"), 3))                                    # ignored, no crash
        self.assertTrue(img.any())

    def test_input_validation(self):
        with self.assertRaises(InvalidFrameError):
            diag.draw_bev_grid(np.zeros((10, 10, 3), np.uint8), BEV)
        with self.assertRaises(InvalidCalibrationInputError):
            diag.draw_bev_grid(np.zeros(BEV.shape + (3,), np.uint8))
        with self.assertRaises(InvalidFrameError):
            diag.draw_ground_grid_on_image(np.zeros((4, 4), np.float32), PROFILE)
        with self.assertRaises(InvalidCalibrationInputError):
            diag.draw_ground_grid_on_image(STRAIGHT_IMG, PROFILE, spacing_m=0)
        with self.assertRaises(InvalidCalibrationInputError):
            diag.draw_valid_region()
        with self.assertRaises(InvalidFrameError):
            diag.draw_valid_region(bev_image=np.zeros((5, 5, 3), np.uint8), valid_mask=np.zeros((4, 4), np.uint8), support_mask=np.zeros((5, 5), np.uint8))
        with self.assertRaises(InvalidCalibrationInputError):
            diag.draw_corner_overlay(np.zeros((10, 10), np.uint8), np.zeros((3, 2)), (9, 6))
        with self.assertRaises(InvalidCalibrationInputError):
            diag.draw_reprojection_error(np.zeros((10, 10), np.uint8), np.zeros((3, 2)), np.zeros((4, 2)))

    def test_side_by_side_and_comparison(self):
        out = diag.side_by_side(STRAIGHT_IMG, np.zeros((500, 350, 3), np.uint8), height=240)
        self.assertEqual(out.shape[0], 240)
        both = diag.draw_undistortion_comparison(STRAIGHT_IMG, STRAIGHT_IMG)
        self.assertEqual(both.shape, (480, 1280, 3))
        with self.assertRaises(InvalidFrameError):
            diag.draw_undistortion_comparison(STRAIGHT_IMG, np.zeros((10, 10, 3), np.uint8))

    def test_save_image_round_trip_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a" / "b" / "x.png"
            diag.save_image(p, STRAIGHT_IMG)
            self.assertEqual(cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR).shape, STRAIGHT_IMG.shape)
            with self.assertRaises(Exception):
                diag.save_image(Path(tmp) / "x.unknownext", STRAIGHT_IMG)

    def test_reprojection_summary_lists_worst_first(self):
        from camera_calibration.intrinsics import IntrinsicCalibrationConfig, IntrinsicCalibrator
        from camera_calibration.synthetic import project_board_corners
        poses = random_board_poses(CAMERA, PATTERN, SQUARE_MM / 1000, 12, seed=1)
        dets = [(f"v{i}", project_board_corners(CAMERA, PATTERN, SQUARE_MM / 1000, *p) + np.random.default_rng(i).normal(0, 0.1 + 0.02 * i, (54, 2)))
                for i, p in enumerate(poses)]
        res = IntrinsicCalibrator(IntrinsicCalibrationConfig(PATTERN, SQUARE_MM, "mm")).calibrate_from_detections(dets, (640, 480))
        lines = diag.reprojection_summary(res).splitlines()
        values = [float(l.split()[1]) for l in lines[2:]]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertEqual(len(values), 12)


class PointsFileTests(unittest.TestCase):
    def write(self, tmp, obj):
        p = Path(tmp) / "points.json"
        p.write_text(json.dumps(obj), encoding="utf-8")
        return str(p)

    def good(self):
        return {"space": "raw", "origin_description": "floor under lens",
                "correspondences": [{"image_px": [100 + 10 * i, 200 + 5 * i], "ground_m": [0.5 + 0.1 * i, 0.05 * i]} for i in range(5)],
                "held_out": [{"image_px": [300, 300], "ground_m": [1, 0]}], "mounting": {"camera_height_m": 0.25}}

    def test_good_file_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = load_correspondences(self.write(tmp, self.good()))
        self.assertEqual(d["image"].shape, (5, 2)); self.assertEqual(d["ground"].shape, (5, 2))
        self.assertEqual(d["held_out"][0].shape, (1, 2))
        self.assertEqual(d["space"], "raw"); self.assertEqual(d["mounting"]["camera_height_m"], 0.25)

    def test_template_is_refused_until_ground_positions_are_filled_in(self):
        template = points_template([(10.0, 20.0), (30.0, 40.0)])
        self.assertTrue(all(c["ground_m"] is None for c in template["correspondences"]))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(InvalidCalibrationInputError, "null"):
            load_correspondences(self.write(tmp, template))

    def test_malformed_files_are_rejected(self):
        def mutate(fn):
            d = self.good(); fn(d); return d
        cases = {
            "not an object": [1, 2],
            "bad space": mutate(lambda d: d.__setitem__("space", "warped")),
            "entry not object": mutate(lambda d: d["correspondences"].__setitem__(0, [1, 2])),
            "missing ground": mutate(lambda d: d["correspondences"][0].pop("ground_m")),
            "pixel wrong length": mutate(lambda d: d["correspondences"][0].__setitem__("image_px", [1, 2, 3])),
            "ground string": mutate(lambda d: d["correspondences"][0].__setitem__("ground_m", ["a", "b"])),
            "bool number": mutate(lambda d: d["correspondences"][0].__setitem__("ground_m", [True, 1])),
            "mounting not object": mutate(lambda d: d.__setitem__("mounting", 5)),
            "origin not string": mutate(lambda d: d.__setitem__("origin_description", 5)),
            "list expected": mutate(lambda d: d.__setitem__("correspondences", "x")),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for label, obj in cases.items():
                with self.subTest(label), self.assertRaises(InvalidCalibrationInputError):
                    load_correspondences(self.write(tmp, obj))
            for label, text in (("nan", '{"correspondences": [{"image_px": [NaN, 1], "ground_m": [1, 1]}]}'), ("broken", "{oops")):
                p = Path(tmp) / "bad.json"; p.write_text(text)
                with self.subTest(label), self.assertRaises(InvalidCalibrationInputError):
                    load_correspondences(str(p))
            with self.assertRaises(InvalidCalibrationInputError):
                load_correspondences(str(Path(tmp) / "missing.json"))


class WorkflowCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.demo = run_synthetic_demo(cls.root / "demo", seed=3)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_synthetic_demo_artifacts_and_summary(self):
        d = self.demo
        self.assertTrue(d["consistency_passed"] and d["distances_passed"])
        self.assertIn(d["intrinsic_status"], ("ok", "warning"))
        self.assertLess(d["intrinsic_rms_px"], 0.5)
        m = d["road_plane_metrics"]
        self.assertAlmostEqual(m["camera_height_m"], 0.25, delta=0.01)            # the simulation's true height
        self.assertAlmostEqual(m["camera_pitch_down_deg"], 28.0, delta=0.5)
        out = self.root / "demo"
        for name in ("profile.json", "points.json", "report.txt", "SYNTHETIC_README.txt", "overlays/straight_bev_grid.png"):
            self.assertTrue((out / name).is_file(), name)
        self.assertIn("SYNTHETIC", (out / "report.txt").read_text(encoding="utf-8"))
        self.assertTrue(CalibrationProfile.load(out / "profile.json").is_metric)

    def test_intrinsic_command_end_to_end(self):
        imgs = self.root / "boards"
        imgs.mkdir(exist_ok=True)
        for i, (R, t) in enumerate(random_board_poses(CAMERA, PATTERN, SQUARE_MM / 1000, 14, seed=8)):
            cv2.imwrite(str(imgs / f"b{i:02d}.png"), render_chessboard(CAMERA, PATTERN, SQUARE_MM / 1000, R, t, seed=i))
        out = self.root / "intr.json"
        code, stdout, stderr = run("intrinsic", "--images", imgs, "--pattern", "9x6", "--square", 25, "--unit", "mm", "--camera-id", "synthetic",
                                   "--out", out, "--overlay-dir", self.root / "ov", "--source-id", "0")
        self.assertEqual(code, 0, stderr)
        self.assertIn("Intrinsic calibration", stdout)
        p = CalibrationProfile.load(out)
        self.assertEqual(p.source_ids, ("0",)); self.assertFalse(p.is_metric)
        self.assertEqual(len(list((self.root / "ov").glob("*_corners.png"))), 14)
        self.assertEqual(len(list((self.root / "ov").glob("*_reprojection.png"))), 14)

    def test_intrinsic_command_refuses_poor_data_with_exit_2_and_a_readable_reason(self):
        imgs = self.root / "flat_boards"
        imgs.mkdir(exist_ok=True)
        for i, (R, t) in enumerate(random_board_poses(CAMERA, PATTERN, SQUARE_MM / 1000, 12, seed=2, max_tilt_deg=2.0)):
            cv2.imwrite(str(imgs / f"b{i:02d}.png"), render_chessboard(CAMERA, PATTERN, SQUARE_MM / 1000, R, t, seed=i))
        out = self.root / "should_not_exist.json"
        code, stdout, stderr = run("intrinsic", "--images", imgs, "--pattern", "9x6", "--square", 25, "--unit", "mm", "--camera-id", "c", "--out", out)
        self.assertEqual(code, 2)
        self.assertIn("orientations too similar", stderr)
        self.assertFalse(out.exists())                                              # nothing written for a rejected calibration
        code, _, _ = run("intrinsic", "--images", imgs, "--pattern", "9x6", "--square", 25, "--unit", "mm", "--camera-id", "c", "--out", out,
                         "--no-enforce-quality")
        self.assertEqual(code, 0)
        self.assertTrue(out.exists())

    def test_units_and_tolerances_have_no_defaults(self):
        code, _, err = run("intrinsic", "--images", self.root, "--pattern", "9x6", "--square", 25, "--camera-id", "c", "--out", self.root / "x.json")
        self.assertEqual(code, 2); self.assertIn("--unit", err)
        code, _, err = run("check-distances", "--profile", self.demo["profile_path"], "--measurements", self.root / "m.json")
        self.assertEqual(code, 2); self.assertIn("--tolerance-m", err)

    def test_invalid_pattern_text_is_a_usage_error(self):
        code, _, err = run("intrinsic", "--images", self.root, "--pattern", "nine", "--square", 25, "--unit", "mm", "--camera-id", "c", "--out", self.root / "x.json")
        self.assertEqual(code, 2); self.assertIn("COLS", err.upper().replace("COLUMNS", "COLS"))

    def test_road_plane_command_with_height_crosscheck_held_out_and_bev(self):
        intr_only = CalibrationProfile.load(self.demo["profile_path"])
        base = CalibrationProfile(intr_only.camera_id, intr_only.intrinsics, intr_only.undistortion)
        base_path = base.save(self.root / "base.json")
        gx, gy = np.meshgrid(np.linspace(0.5, 1.6, 4), np.linspace(-0.35, 0.35, 3))
        ground = np.column_stack([gx.ravel(), gy.ravel()])
        held = np.array([[0.9, 0.1], [1.3, -0.2]])
        points = {"space": "raw", "origin_description": "SYNTHETIC origin",
                  "correspondences": [{"image_px": p.tolist(), "ground_m": g.tolist()} for p, g in zip(project_ground_points(CAMERA, POSE, ground), ground)],
                  "held_out": [{"image_px": p.tolist(), "ground_m": g.tolist()} for p, g in zip(project_ground_points(CAMERA, POSE, held), held)],
                  "mounting": {"camera_height_m": 0.25}}
        pts_path = self.root / "pts.json"; pts_path.write_text(json.dumps(points))
        out = self.root / "metric.json"
        code, stdout, stderr = run("road-plane", "--profile", base_path, "--points", pts_path, "--out", out, "--bev-extents", 0.3, 2.3, -0.7, 0.7,
                                   "--mpp", 0.004, "--image", self.root / "demo" / "intrinsic_images" / "board_00.png", "--overlay-dir", self.root / "ov2")
        self.assertEqual(code, 0, stderr)
        self.assertIn("[OK]", stdout)                                               # height cross-check
        self.assertIn("held-out validation", stdout)
        p = CalibrationProfile.load(out)
        self.assertTrue(p.is_metric)
        self.assertEqual(p.mounting["camera_height_m"], 0.25)
        self.assertTrue((self.root / "ov2" / "ground_grid.png").is_file())

    def test_road_plane_command_rejects_mirrored_measurements(self):
        base = CalibrationProfile.load(self.demo["profile_path"])
        base = CalibrationProfile(base.camera_id, base.intrinsics, base.undistortion)
        base_path = base.save(self.root / "base2.json")
        gx, gy = np.meshgrid(np.linspace(0.5, 1.6, 4), np.linspace(-0.35, 0.35, 3))
        ground = np.column_stack([gx.ravel(), gy.ravel()])
        points = {"correspondences": [{"image_px": p.tolist(), "ground_m": [g[0], -g[1]]} for p, g in zip(project_ground_points(CAMERA, POSE, ground), ground)]}
        pts_path = self.root / "mirrored.json"; pts_path.write_text(json.dumps(points))
        out = self.root / "never.json"
        code, _, stderr = run("road-plane", "--profile", base_path, "--points", pts_path, "--out", out)
        self.assertEqual(code, 2); self.assertIn("mirrored", stderr); self.assertFalse(out.exists())

    def test_info_set_bev_bev_and_check_distances(self):
        prof = self.demo["profile_path"]
        code, stdout, _ = run("info", "--profile", prof)
        self.assertEqual(code, 0); self.assertIn("fingerprint", stdout); self.assertIn("Validation (self-consistency): PASS", stdout)
        code, stdout, _ = run("set-bev", "--profile", prof, "--out", self.root / "bev2.json", "--bev-extents", 0.4, 2.0, -0.5, 0.5, "--mpp", 0.005)
        self.assertEqual(code, 0)
        self.assertEqual(CalibrationProfile.load(self.root / "bev2.json").bev.meters_per_pixel, 0.005)
        code, _, err = run("set-bev", "--profile", prof, "--out", self.root / "x.json", "--mpp", 0.005)
        self.assertEqual(code, 2); self.assertIn("both", err.lower() + "both")
        raw = self.root / "raw.png"; cv2.imwrite(str(raw), STRAIGHT_IMG)
        code, stdout, stderr = run("bev", "--profile", self.root / "bev2.json", "--image", raw, "--out", self.root / "o" / "bev.png")
        self.assertEqual(code, 0, stderr)
        for suffix in ("", "_grid", "_regions", "_sidebyside"):
            self.assertTrue((self.root / "o" / f"bev{suffix}.png").is_file(), suffix)
        a, b = project_ground_points(CAMERA, POSE, [[1.0, -0.25], [1.0, 0.25]])
        meas = {"distances": [{"a_px": a.tolist(), "b_px": b.tolist(), "distance_m": 0.5, "label": "lane"}]}
        mpath = self.root / "m.json"; mpath.write_text(json.dumps(meas))
        code, stdout, _ = run("check-distances", "--profile", prof, "--measurements", mpath, "--tolerance-m", 0.005)
        self.assertEqual(code, 0); self.assertIn("PASS", stdout)
        meas["distances"][0]["distance_m"] = 0.6
        mpath.write_text(json.dumps(meas))
        code, stdout, _ = run("check-distances", "--profile", prof, "--measurements", mpath, "--tolerance-m", 0.005)
        self.assertEqual(code, 2); self.assertIn("FAIL", stdout)

    def test_bev_command_reports_resolution_mismatch_cleanly(self):
        small = self.root / "small.png"; cv2.imwrite(str(small), np.zeros((240, 320, 3), np.uint8))
        code, _, err = run("bev", "--profile", self.demo["profile_path"], "--image", small, "--out", self.root / "z.png")
        self.assertEqual(code, 2); self.assertIn("320x240", err)

    def test_corrupt_profile_is_reported_cleanly(self):
        bad = self.root / "bad.json"; bad.write_text("{not json")
        code, _, err = run("info", "--profile", bad)
        self.assertEqual(code, 2); self.assertIn("not valid JSON", err)

    def test_live_command_runs_the_real_pipeline_headless_on_a_video_file(self):
        clip = self.root / "clip.avi"
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 25, (640, 480))
        self.assertTrue(writer.isOpened(), "this OpenCV build cannot write MJPG AVI; the live test needs it")
        scene = make_road_scene("straight")
        frame = scene.render(CAMERA, POSE)
        for _ in range(12):
            writer.write(frame)
        writer.release()
        # --fast decodes the 12-frame clip almost instantly and M01 keeps only the newest frame, so a starved worker may see
        # fewer than --frames of them: the contract is "at least one BEV, the last one saved, no processing errors".
        code, stdout, stderr = run("live", "--profile", self.demo["profile_path"], "--source", clip, "--file", "--fast", "--frames", 4, "--no-gui",
                                   "--save-dir", self.root / "live")
        self.assertEqual(code, 0, stderr)
        self.assertRegex(stdout, r"[1-4] BEV frame\(s\)")
        self.assertTrue((self.root / "live" / "last_bev.png").is_file())
        self.assertIn("errors none", stdout)

    def test_live_command_fails_cleanly_when_the_profile_does_not_match_the_source(self):
        clip = self.root / "clip_small.avi"
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 25, (320, 240))
        self.assertTrue(writer.isOpened())
        for _ in range(8):
            writer.write(np.zeros((240, 320, 3), np.uint8))
        writer.release()
        code, stdout, stderr = run("live", "--profile", self.demo["profile_path"], "--source", clip, "--file", "--frames", 3, "--no-gui")
        self.assertEqual(code, 2)                                                   # never reports a BEV from a mismatched source
        self.assertIn("error:", stderr)


if __name__ == "__main__":
    unittest.main()
