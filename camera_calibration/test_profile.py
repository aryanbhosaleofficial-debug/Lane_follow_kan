"""Tests for calibration profile management (SYNTHETIC data)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from camera_calibration.bev import BEVConfig
from camera_calibration.errors import (InvalidBEVRegionError, ProfileError, ResolutionMismatchError, SourceMismatchError)
from camera_calibration.profile import MAX_PROFILE_BYTES, CalibrationProfile, new_profile
from camera_calibration.roadplane import RoadPlaneCalibration, apply_homography, convex_hull
from camera_calibration.synthetic import SyntheticCamera, SyntheticPose, true_ground_homography

CAM = SyntheticCamera(dist=(-0.2, 0.05, 0.001, -0.0005, 0.0))
POSE = SyntheticPose(height_m=0.25, pitch_down_deg=28.0)
GROUND = np.array([[0.5, -0.3], [0.5, 0.3], [1.5, -0.3], [1.5, 0.3], [1.0, 0.0], [2.0, 0.0]])


def make_road_plane(profile: CalibrationProfile, **overrides) -> RoadPlaneCalibration:
    """Build a road plane directly from the exact synthetic homography (no estimator involved)."""
    Kn = profile.undistortion.camera_matrix
    H = true_ground_homography(POSE, Kn)
    und, valid = apply_homography(H, GROUND)
    assert valid.all()
    kw = dict(homography=H, undistorted_camera_matrix=Kn, image_size=profile.image_size, ground_points_m=GROUND,
              image_points_undistorted_px=und, image_points_raw_px=None, inlier_mask=np.ones(len(GROUND), bool),
              method="synthetic", camera_facing="forward", origin_description="SYNTHETIC origin",
              calibrated_hull_m=convex_hull(GROUND), metrics={"max_reprojection_px": 0.0, "rms_reprojection_px": 0.0},
              exact_fit=False)
    kw.update(overrides)
    return RoadPlaneCalibration(**kw)


def full_profile(**kw) -> CalibrationProfile:
    p = new_profile(CAM.intrinsics(), "synthetic-cam", camera_note="SYNTHETIC", **kw)
    p = p.with_road_plane(make_road_plane(p))
    return p.with_bev(BEVConfig.from_extents(0.3, 2.3, -0.8, 0.8, 0.005))


class ProfileRoundTripTests(unittest.TestCase):
    def test_save_load_preserves_everything(self):
        p = full_profile(source_ids=("0",)).with_mounting(camera_height_m=0.25, notes="SYNTHETIC")
        with tempfile.TemporaryDirectory() as tmp:
            path = p.save(Path(tmp) / "sub" / "profile.json")
            q = CalibrationProfile.load(path)
        self.assertEqual(q.camera_id, "synthetic-cam")
        self.assertEqual(q.source_ids, ("0",))
        self.assertEqual(q.mounting["camera_height_m"], 0.25)
        np.testing.assert_allclose(q.intrinsics.camera_matrix, p.intrinsics.camera_matrix, atol=0, rtol=1e-15)
        np.testing.assert_allclose(q.road_plane.homography, p.road_plane.homography, atol=0, rtol=1e-15)
        np.testing.assert_allclose(q.undistortion.camera_matrix, p.undistortion.camera_matrix)
        self.assertEqual(q.bev, p.bev)
        self.assertEqual(q.fingerprint(), p.fingerprint())
        self.assertTrue(q.is_metric)

    def test_intrinsics_only_profile_round_trips(self):
        p = new_profile(CAM.intrinsics(), "intrinsics-only")
        self.assertFalse(p.is_metric)
        with tempfile.TemporaryDirectory() as tmp:
            q = CalibrationProfile.load(p.save(Path(tmp) / "p.json"))
        self.assertIsNone(q.road_plane)
        with self.assertRaisesRegex(ProfileError, "no road-plane"):
            q.require_metric()

    def test_fingerprint_tracks_geometry_not_notes(self):
        p = full_profile()
        self.assertEqual(p.fingerprint(), p.with_mounting(extra="note").fingerprint())
        other = p.with_bev(BEVConfig.from_extents(0.3, 2.3, -0.8, 0.8, 0.01))
        self.assertNotEqual(p.fingerprint(), other.fingerprint())

    def test_saved_file_is_plain_json_with_no_nonfinite_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = full_profile().save(Path(tmp) / "p.json").read_text(encoding="utf-8")
        data = json.loads(text, parse_constant=lambda c: self.fail(f"non-finite constant {c} written"))
        self.assertEqual(data["format"], "m02-calibration-profile")
        self.assertEqual(data["road_plane"]["frame"], "X forward, Y left, Z up; metres")

    def test_failed_save_leaves_existing_file_untouched_and_no_temp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            full_profile().save(path)
            before = path.read_bytes()
            with patch("camera_calibration.profile.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    full_profile().save(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([f.name for f in Path(tmp).iterdir()], ["p.json"])


class MalformedProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.good = full_profile().to_dict()

    def load_text(self, text, raw=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            path.write_bytes(raw if raw is not None else text.encode("utf-8"))
            return CalibrationProfile.load(path)

    def mutated(self, fn):
        d = copy.deepcopy(self.good)
        fn(d)
        return d

    def test_file_level_problems(self):
        cases = {
            "empty": "",
            "not json": "{this is not json",
            "array": "[1, 2, 3]",
            "NaN constant": '{"a": NaN}',
            "Infinity constant": '{"a": Infinity}',
            "duplicate keys": '{"format": "x", "format": "y"}',
            "truncated": json.dumps(self.good)[:200],
        }
        for label, text in cases.items():
            with self.subTest(label), self.assertRaises(ProfileError):
                self.load_text(text)

    def test_bad_bytes_and_size_limits(self):
        with self.assertRaisesRegex(ProfileError, "UTF-8"):
            self.load_text("", raw=b"\xff\xfe\x00bad")
        with self.assertRaisesRegex(ProfileError, "byte limit"):
            self.load_text("", raw=b" " * (MAX_PROFILE_BYTES + 1))
        with self.assertRaisesRegex(ProfileError, "not found"):
            CalibrationProfile.load("no/such/profile.json")
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ProfileError):
            CalibrationProfile.load(tmp)                    # a directory is not a profile

    def test_deeply_nested_json_is_a_profile_error_not_a_crash(self):
        # 200 KB, far below the size limit, but it overflows the JSON decoder's stack (RecursionError, not ValueError)
        with self.assertRaisesRegex(ProfileError, "nested too deeply"):
            self.load_text("", raw=b"[" * 100_000 + b"]" * 100_000)

    def test_axis_convention_markers_are_checked_on_load(self):
        d = self.mutated(lambda d: d["road_plane"].__setitem__("homography_maps", "raw pixels -> ground"))
        with self.assertRaisesRegex(ProfileError, "homography_maps"):
            CalibrationProfile.from_dict(d)
        d = self.mutated(lambda d: d["road_plane"].__setitem__("frame", "X right, Y forward, Z down; metres"))
        with self.assertRaisesRegex(ProfileError, "road_plane.frame"):
            CalibrationProfile.from_dict(d)
        d = self.mutated(lambda d: d["bev"].__setitem__("convention", "u left, v up"))
        with self.assertRaisesRegex(ProfileError, "bev.convention"):
            CalibrationProfile.from_dict(d)
        self.assertEqual(CalibrationProfile.from_dict(copy.deepcopy(self.good)).fingerprint(),
                         CalibrationProfile.from_dict(self.good).fingerprint())    # an untouched profile still loads

    def test_non_finite_values_hidden_in_valid_json_are_rejected(self):
        d = self.mutated(lambda d: d["intrinsics"]["camera_matrix"][0].__setitem__(0, 1e999))   # becomes inf on load
        with self.assertRaises(ProfileError):
            self.load_text(json.dumps(d).replace("1e+999", "1e999"))

    def test_structural_mutations_are_rejected(self):
        def drop(path):
            def f(d):
                node = d
                for k in path[:-1]:
                    node = node[k]
                del node[path[-1]]
            return f

        def setv(path, value):
            def f(d):
                node = d
                for k in path[:-1]:
                    node = node[k]
                node[path[-1]] = value
            return f
        cases = {
            "wrong format": setv(["format"], "something-else"),
            "future schema": setv(["schema_version"], 99),
            "bool schema": setv(["schema_version"], True),
            "missing camera_id": drop(["camera_id"]),
            "empty camera_id": setv(["camera_id"], "  "),
            "missing intrinsics": drop(["intrinsics"]),
            "missing camera_matrix": drop(["intrinsics", "camera_matrix"]),
            "camera_matrix 2x2": setv(["intrinsics", "camera_matrix"], [[1, 0], [0, 1]]),
            "camera_matrix strings": setv(["intrinsics", "camera_matrix"], [["a", "b", "c"]] * 3),
            "dist wrong length": setv(["intrinsics", "dist_coeffs"], [0.0, 0.0]),
            "unknown model": setv(["intrinsics", "model"], "magic"),
            "principal point outside": setv(["intrinsics", "camera_matrix", 0, 2], 9999.0),
            "negative focal": setv(["intrinsics", "camera_matrix", 0, 0], -5.0),
            "size zero": setv(["intrinsics", "image_width"], 0),
            "undistortion missing": drop(["undistortion"]),
            "undistortion size mismatch": setv(["undistortion", "output_size"], [320, 240]),
            "undistortion K bad": setv(["undistortion", "camera_matrix"], [[1, 2, 3]]),
            "road plane not object": setv(["road_plane"], [1, 2]),
            "road plane missing H": drop(["road_plane", "homography"]),
            "road plane H wrong shape": setv(["road_plane", "homography"], [[1, 0, 0], [0, 1, 0]]),
            "road plane H not normalised": setv(["road_plane", "homography"], [[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
            "road plane K differs": setv(["road_plane", "undistorted_camera_matrix", 0, 0], 700.0),
            "road plane size differs": setv(["road_plane", "image_size"], [320, 240]),
            "road plane mask wrong type": setv(["road_plane", "inlier_mask"], [1, 1, 1, 1, 1, 1]),
            "road plane mask too small": setv(["road_plane", "inlier_mask"], [True, True, False, False, False, False]),
            "road plane bad facing": setv(["road_plane", "camera_facing"], "sideways"),
            "road plane tampered homography": setv(["road_plane", "homography", 0, 0], 0.5),
            "road plane tampered points": setv(["road_plane", "image_points_undistorted_px", 0, 0], 5.0),
            "road plane metric nan-like string": setv(["road_plane", "metrics", "rms_reprojection_px"], "bad"),
            "bev missing field": drop(["bev", "meters_per_pixel"]),
            "bev zero scale": setv(["bev", "meters_per_pixel"], 0),
            "bev negative size": setv(["bev", "width_px"], -4),
            "bev huge": setv(["bev", "width_px"], 100000),
            "bev bad interpolation": setv(["bev", "interpolation"], "sinc"),
            "bev without road plane": lambda d: (d.__setitem__("road_plane", None)),
            "credentials in camera_id": setv(["camera_id"], "rtsp://admin:hunter2@10.0.0.2/stream"),
            "credentials in source_ids": setv(["source_ids"], ["rtsp://admin:hunter2@10.0.0.2/stream"]),
            "credentials in mounting": setv(["mounting"], {"url": "rtsp://u:p@host/x"}),
            "bad created_utc": setv(["created_utc"], "yesterday"),
            "mounting not object": setv(["mounting"], [1]),
            "mounting nested too deep": setv(["mounting"], {"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}}),
        }
        for label, fn in cases.items():
            with self.subTest(label), self.assertRaises((ProfileError, InvalidBEVRegionError)):
                CalibrationProfile.from_dict(self.mutated(fn))

    def test_unmutated_dict_loads(self):
        self.assertTrue(CalibrationProfile.from_dict(copy.deepcopy(self.good)).is_metric)

    def test_credentials_never_written(self):
        for leaky in ("rtsp://admin:secret@host/s", "rtsp://admin@host/s", "http://u:p@h"):
            with self.subTest(leaky), self.assertRaises(ProfileError):
                new_profile(CAM.intrinsics(), "cam", source_ids=(leaky,))
        # M01 already redacts credentials in FramePacket.source_id; that redacted form must stay bindable
        self.assertEqual(new_profile(CAM.intrinsics(), "cam", source_ids=("rtsp://***@host/s",)).source_ids,
                         ("rtsp://***@host/s",))


class CompatibilityPolicyTests(unittest.TestCase):
    def test_resolution_must_match_exactly(self):
        p = full_profile()
        p.check_compatibility(640, 480)
        for size in ((320, 240), (1280, 720), (640, 481), (480, 640)):
            with self.subTest(size=size), self.assertRaises(ResolutionMismatchError):
                p.check_compatibility(*size)

    def test_source_binding(self):
        p = full_profile(source_ids=("0", "usb-cam"))
        p.check_compatibility(640, 480, "0")
        p.check_compatibility(640, 480, "usb-cam")
        with self.assertRaises(SourceMismatchError):
            p.check_compatibility(640, 480, "1")
        with self.assertRaises(SourceMismatchError):
            p.check_compatibility(640, 480, None)
        p.check_compatibility(640, 480, "1", strict_source=False)

    def test_unbound_profile_accepts_any_source(self):
        full_profile().check_compatibility(640, 480, "anything")

    def test_alpha_change_after_road_plane_is_refused(self):
        with self.assertRaisesRegex(ProfileError, "invalidates"):
            full_profile().with_undistortion_alpha(1.0)
        changed = new_profile(CAM.intrinsics(), "c").with_undistortion_alpha(1.0)
        self.assertEqual(changed.undistortion.alpha, 1.0)

    def test_mismatched_undistortion_is_caught_by_profile_constructor(self):
        p = full_profile()
        other = new_profile(CAM.intrinsics(), "c", alpha=1.0).undistortion
        with self.assertRaisesRegex(ProfileError, "different undistorted camera matrix"):
            CalibrationProfile(p.camera_id, p.intrinsics, other, p.road_plane, p.bev)


class ScalingTests(unittest.TestCase):
    def test_scaled_profile_projects_consistently(self):
        p = full_profile()
        half = p.scaled_to(320, 240)
        self.assertEqual(half.image_size, (320, 240))
        self.assertIn("rescaled", " ".join(half.road_plane.warnings))
        # ground point -> undistorted pixel: half-size result is the full-size result scaled (pixel-centre convention)
        full_px, _ = apply_homography(p.road_plane.homography, GROUND)
        half_px, valid = apply_homography(half.road_plane.homography, GROUND)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(half_px, 0.5 * (full_px + 0.5) - 0.5, atol=1e-9)
        half.check_compatibility(320, 240)

    def test_scaling_refuses_aspect_change(self):
        with self.assertRaisesRegex(Exception, "aspect ratio"):
            full_profile().scaled_to(640, 360)

    def test_scaled_profile_round_trips_through_json(self):
        half = full_profile().scaled_to(320, 240)
        with tempfile.TemporaryDirectory() as tmp:
            q = CalibrationProfile.load(half.save(Path(tmp) / "p.json"))
        self.assertEqual(q.image_size, (320, 240))


if __name__ == "__main__":
    unittest.main()
