"""Tests for the BEV grid, metric conversions and the fused BEV remap (SYNTHETIC data, exact ground truth)."""
import threading
import unittest

import cv2
import numpy as np

from camera_calibration import mapper as mapper_mod
from camera_calibration.bev import MAX_BEV_PIXELS, BEVConfig
from camera_calibration.errors import (DegenerateGeometryError, InvalidBEVRegionError, InvalidFrameError, ProfileError,
                                       ResolutionMismatchError)
from camera_calibration.mapper import BEVMapper, CameraGeometry
from camera_calibration.plane_calibration import RoadPlaneCalibrator, RoadPlaneConfig
from camera_calibration.profile import new_profile
from camera_calibration.synthetic import (SyntheticCamera, SyntheticPose, make_road_scene, project_ground_points)

CAM = SyntheticCamera(dist=(-0.28, 0.09, 0.001, -0.0008, -0.01))
FISH = SyntheticCamera(fx=330.0, fy=330.0, dist=(0.04, -0.01, 0.002, -0.0005), model="fisheye")
POSE = SyntheticPose(height_m=0.25, pitch_down_deg=28.0)
GX, GY = np.meshgrid(np.linspace(0.5, 1.6, 4), np.linspace(-0.35, 0.35, 3))
GROUND = np.column_stack([GX.ravel(), GY.ravel()])
MPP = 0.004
BEV = BEVConfig.from_extents(0.3, 2.3, -0.7, 0.7, MPP)
ORIGIN_U, ORIGIN_V = BEV.origin_u_px, BEV.origin_v_px


def build_profile(cam=CAM, pose=POSE, bev=BEV, noise=0.0, seed=0):
    prof = new_profile(cam.intrinsics(), "synthetic")
    px = project_ground_points(cam, pose, GROUND)
    if noise:
        px = px + np.random.default_rng(seed).normal(0, noise, px.shape)
    fit = RoadPlaneCalibrator(prof.intrinsics, prof.undistortion, RoadPlaneConfig(origin_description="SYNTHETIC")).estimate(px, GROUND)
    prof = prof.with_road_plane(fit)
    return prof.with_bev(bev) if bev is not None else prof


PROFILE = build_profile()
GEO = CameraGeometry(PROFILE)
STRAIGHT_IMG = make_road_scene("straight").render(CAM, POSE)
CURVE_IMG = make_road_scene("curve").render(CAM, POSE)


# ---------------------------------------------------------------------------------------------
class BEVGridTests(unittest.TestCase):
    def test_from_extents_geometry(self):
        cfg = BEVConfig.from_extents(0.3, 2.3, -0.7, 0.7, 0.004)
        self.assertEqual(cfg.shape, (500, 350))
        self.assertAlmostEqual(cfg.x_max_m, 2.3, places=12)
        self.assertAlmostEqual(cfg.y_max_m, 0.7, places=12)
        self.assertAlmostEqual(cfg.x_min_m, 0.3, places=9)
        self.assertAlmostEqual(cfg.y_min_m, -0.7, places=9)

    def test_axis_orientation_is_explicit(self):
        # forward is UP in the image, the vehicle's left is the image's LEFT
        o = BEV.ground_to_bev([[0.0, 0.0]])[0]
        np.testing.assert_allclose(o, [ORIGIN_U, ORIGIN_V])
        fwd = BEV.ground_to_bev([[1.0, 0.0]])[0]
        left = BEV.ground_to_bev([[0.0, 1.0]])[0]
        self.assertLess(fwd[1], o[1]); self.assertEqual(fwd[0], o[0])        # +x -> smaller v (up), same column
        self.assertLess(left[0], o[0]); self.assertEqual(left[1], o[1])      # +y -> smaller u (left), same row
        self.assertAlmostEqual(o[1] - fwd[1], 1.0 / 0.004)                    # exact scale: 1 m = 250 px

    def test_pixel_centre_convention(self):
        # the centre of the top-left pixel is the ground point half a pixel inside the top-left corner
        np.testing.assert_allclose(BEV.bev_to_ground([[0, 0]])[0], [BEV.x_max_m - 0.5 * MPP, BEV.y_max_m - 0.5 * MPP])
        np.testing.assert_allclose(BEV.bev_to_ground([[349, 499]])[0], [BEV.x_min_m + 0.5 * MPP, BEV.y_min_m + 0.5 * MPP], atol=1e-12)

    def test_round_trip_is_exact_and_matches_matrix_form(self):
        rng = np.random.default_rng(0)
        g = rng.uniform(-3, 3, (200, 2))
        np.testing.assert_allclose(BEV.bev_to_ground(BEV.ground_to_bev(g)), g, atol=1e-12)
        homog = np.column_stack([g, np.ones(200)]) @ BEV.ground_to_bev_matrix().T
        np.testing.assert_allclose(homog[:, :2], BEV.ground_to_bev(g), atol=1e-9)
        np.testing.assert_allclose(homog[:, 2], 1.0)

    def test_nan_propagates_and_batch_shapes(self):
        out = BEV.ground_to_bev([[1.0, 0.0], [np.nan, 0.0]])
        self.assertTrue(np.isfinite(out[0]).all()); self.assertTrue(np.isnan(out[1]).all())
        self.assertEqual(BEV.ground_to_bev([1.0, 0.0]).shape, (1, 2))
        self.assertEqual(BEV.ground_to_bev(np.zeros((0, 2))).shape, (0, 2))

    def test_contains(self):
        inside = BEV.contains_bev([[0, 0], [349, 499], [-0.5, -0.5], [350, 100], [-1, 5], [np.nan, 1]])
        self.assertEqual(inside.tolist(), [True, True, True, False, False, False])

    def test_invalid_configs(self):
        base = dict(width_px=100, height_px=100, meters_per_pixel=0.01, origin_u_px=50, origin_v_px=100)
        cases = {"zero width": dict(width_px=0), "float width": dict(width_px=10.5), "bool height": dict(height_px=True),
                 "zero scale": dict(meters_per_pixel=0), "negative scale": dict(meters_per_pixel=-1),
                 "nan scale": dict(meters_per_pixel=float("nan")), "nan origin": dict(origin_u_px=float("nan")),
                 "inf origin": dict(origin_v_px=float("inf")), "string origin": dict(origin_u_px="3"),
                 "bad interpolation": dict(interpolation="sinc"), "border > 255": dict(border_value=300),
                 "border negative": dict(border_value=-1), "border bool": dict(border_value=True),
                 "negative margin": dict(support_margin_m=-0.1), "too big": dict(width_px=MAX_BEV_PIXELS, height_px=2)}
        for label, over in cases.items():
            with self.subTest(label), self.assertRaises(InvalidBEVRegionError):
                BEVConfig(**{**base, **over})
        self.assertIsInstance(BEVConfig(**base), BEVConfig)

    def test_invalid_extents(self):
        for args in ((1, 1, -1, 1, 0.01), (2, 1, -1, 1, 0.01), (0, 1, 1, -1, 0.01), (0, 1, -1, 1, 0), (0, 1e-4, -1e-4, 1e-4, 0.01),
                     (0, float("nan"), -1, 1, 0.01), (0, 1, -1, 1, float("inf"))):
            with self.subTest(args), self.assertRaises(InvalidBEVRegionError):
                BEVConfig.from_extents(*args)

    def test_dict_round_trip(self):
        self.assertEqual(BEVConfig.from_dict(BEV.to_dict()), BEV)
        with self.assertRaises(InvalidBEVRegionError):
            BEVConfig.from_dict({"width_px": 5})


# ---------------------------------------------------------------------------------------------
class ConversionTests(unittest.TestCase):
    def test_ground_to_image_matches_exact_projection(self):
        pts = np.array([[0.5, -0.3], [1.0, 0.0], [1.6, 0.3], [2.0, 0.1]])
        np.testing.assert_allclose(GEO.ground_to_image(pts), project_ground_points(CAM, POSE, pts), atol=1e-4)

    def test_image_to_ground_inverts_ground_to_image(self):
        rng = np.random.default_rng(1)
        g = np.column_stack([rng.uniform(0.4, 2.5, 300), rng.uniform(-0.6, 0.6, 300)])
        px = GEO.ground_to_image(g)
        keep = GEO.image_contains(px)
        self.assertGreater(keep.sum(), 100)
        np.testing.assert_allclose(GEO.image_to_ground(px[keep]), g[keep], atol=2e-6)
        # the undistorted-space route is the same transformation
        und = GEO.ground_to_image(g, undistorted=True)
        np.testing.assert_allclose(GEO.image_to_ground(und, undistorted=True), g, atol=1e-9)
        np.testing.assert_allclose(GEO.undistort_points(px[keep]), und[keep], atol=1e-6)

    def test_known_reference_lengths(self):
        # two marks 0.50 m apart across the lane and 0.80 m apart along it (exact synthetic data)
        a, b, c = np.array([[1.0, -0.25], [1.0, 0.25], [1.8, 0.25]])
        px = project_ground_points(CAM, POSE, [a, b, c])
        g = GEO.image_to_ground(px)
        self.assertAlmostEqual(float(GEO.ground_distance(g[0:1], g[1:2])[0]), 0.50, places=6)
        self.assertAlmostEqual(float(GEO.ground_distance(g[1:2], g[2:3])[0]), 0.80, places=6)

    def test_reference_lengths_with_realistic_pixel_noise(self):
        noisy = CameraGeometry(build_profile(noise=0.5, seed=2))
        a, b = project_ground_points(CAM, POSE, [[1.0, -0.25], [1.0, 0.25]])
        d = float(noisy.ground_distance(noisy.image_to_ground([a]), noisy.image_to_ground([b]))[0])
        self.assertAlmostEqual(d, 0.50, delta=0.005)         # measured ~2 mm typical for 0.5 px noise inside the hull

    def test_no_x_y_swap_or_sign_error(self):
        # a point ahead and to the LEFT must have x>0 and y>0, and appear left of the image centre
        left_ahead = project_ground_points(CAM, POSE, [[1.2, 0.2]])
        self.assertLess(left_ahead[0, 0], CAM.cx)
        g = GEO.image_to_ground(left_ahead)[0]
        self.assertGreater(g[0], 1.0); self.assertGreater(g[1], 0.0)
        np.testing.assert_allclose(g, [1.2, 0.2], atol=1e-5)
        # a pixel nearer the image bottom is closer to the vehicle
        px = np.array([[CAM.cx, 300.0], [CAM.cx, 400.0]])
        near_far = GEO.image_to_ground(px)
        self.assertGreater(near_far[0, 0], near_far[1, 0])
        # BEV: the same point lands left of and above the origin
        u, v = GEO.ground_to_bev([[1.2, 0.2]])[0]
        self.assertLess(u, ORIGIN_U); self.assertLess(v, ORIGIN_V)

    def test_composites_round_trip(self):
        g = np.array([[0.8, -0.2], [1.5, 0.3]])
        px = GEO.ground_to_image(g)
        np.testing.assert_allclose(GEO.image_to_bev(px), GEO.ground_to_bev(g), atol=1e-5 / MPP)
        np.testing.assert_allclose(GEO.bev_to_image(GEO.image_to_bev(px)), px, atol=1e-4)

    def test_pixels_above_the_horizon_are_invalid(self):
        # With 28 deg of pitch the horizon is out of frame; use a shallow pose where it crosses the image
        # (horizon row = cy - fy*tan(10 deg) = 147.8) so pixels above it can never meet the road ahead.
        geo = CameraGeometry(build_profile(pose=SyntheticPose(0.25, 10.0)))
        above = [[CAM.cx, 0.0], [CAM.cx, 100.0]]
        out, valid = geo.image_to_ground(above, return_valid=True)
        self.assertEqual(valid.tolist(), [False, False])
        self.assertTrue(np.isnan(out).all())
        below, ok = geo.image_to_ground([[CAM.cx, 300.0]], return_valid=True)
        self.assertTrue(ok[0] and np.isfinite(below).all())
        with self.assertRaisesRegex(DegenerateGeometryError, "no valid mapping"):
            geo.image_to_ground(above, on_invalid="raise")

    def test_nonfinite_pixels_are_nan_and_bad_mode_is_rejected(self):
        out = GEO.image_to_ground([[CAM.cx, 400.0], [np.nan, 100.0], [np.inf, 5.0]])
        self.assertTrue(np.isfinite(out[0]).all()); self.assertTrue(np.isnan(out[1:]).all())
        with self.assertRaises(ValueError):
            GEO.image_to_ground([[1, 1]], on_invalid="ignore")

    def test_ground_points_behind_the_camera_are_invalid(self):
        out, valid = GEO.ground_to_image([[-0.5, 0.0], [1.0, 0.0]], return_valid=True)
        self.assertEqual(valid.tolist(), [False, True])
        self.assertTrue(np.isnan(out[0]).all())
        with self.assertRaises(DegenerateGeometryError):
            GEO.ground_to_image([[-0.5, 0.0]], on_invalid="raise")

    def test_batch_and_degenerate_shapes(self):
        self.assertEqual(GEO.image_to_ground(np.zeros((0, 2))).shape, (0, 2))
        self.assertEqual(GEO.image_to_ground([300.0, 400.0]).shape, (1, 2))
        for bad in (np.zeros((4, 3)), "abc", [[1, 2, 3]]):
            with self.subTest(bad=str(bad)[:20]), self.assertRaises(ValueError):
                GEO.image_to_ground(bad)

    def test_region_classification(self):
        pts = [[1.0, 0.0],        # inside hull, visible, inside BEV
               [2.0, 0.0],        # visible and in BEV but outside the calibrated hull (extrapolated)
               [-0.5, 0.0],       # behind the camera
               [1.0, 3.0],        # far to the side: outside the image
               [1.0, 0.9]]        # inside image? outside the BEV laterally
        r = GEO.check_ground_points(pts)
        self.assertEqual(r["in_calibrated_hull"].tolist(), [True, False, False, False, False])
        self.assertEqual(r["in_front"].tolist(), [True, True, False, True, True])
        self.assertEqual(r["lens_valid"].tolist(), [True, True, False, False, True])        # 3 m to the side is beyond the lens model
        self.assertFalse(r["in_image"][2]); self.assertFalse(r["in_image"][3])
        self.assertEqual(r["in_bev"].tolist(), [True, True, False, False, False])
        self.assertTrue(GEO.in_calibrated_hull([[0.45, 0.0]], margin_m=0.1)[0])
        self.assertFalse(GEO.in_calibrated_hull([[0.45, 0.0]], margin_m=0.0)[0])

    def test_ghost_projections_beyond_the_lens_fold_back_are_invalid(self):
        # Regression: a calibrated polynomial (k1..k5 below, from a real synthetic calibration run) turns over near
        # r = 1.3, so near-field BEV corners (undistorted radius ~2x the image half-diagonal) used to project to finite
        # "ghost" pixels INSIDE the image that undistorted onto the wrong branch (0.4 m round-trip error).
        from camera_calibration.intrinsics import IntrinsicParameters
        from camera_calibration.test_profile import make_road_plane
        from camera_calibration.validation import CalibrationValidator
        folding = IntrinsicParameters(640, 480, CAM.camera_matrix(), np.array([-0.2843, 0.1094, 0.001, -0.0009, -0.0323]), "standard5")
        base = new_profile(folding, "folding")
        profile = base.with_road_plane(make_road_plane(base)).with_bev(BEV)
        geo = CameraGeometry(profile)
        corners = np.array([[0.31, 0.66], [0.31, -0.67], [1.0, 0.0]])
        px, valid = geo.ground_to_image(corners, return_valid=True)
        self.assertEqual(valid.tolist(), [False, False, True])                 # ghosts rejected, the genuine point kept
        self.assertTrue(np.isnan(px[:2]).all())
        # every accepted point round-trips exactly
        rng = np.random.default_rng(0)
        g = np.column_stack([rng.uniform(BEV.x_min_m, BEV.x_max_m, 3000), rng.uniform(BEV.y_min_m, BEV.y_max_m, 3000)])
        p, ok = geo.ground_to_image(g, return_valid=True)
        np.testing.assert_allclose(geo.image_to_ground(p[ok]), g[ok], atol=1e-4)           # 0.01 px acceptance => << 0.1 mm
        # the mapper agrees: those BEV pixels are unobserved, and the validator passes
        mapper = BEVMapper(profile)
        u, v = BEV.ground_to_bev(corners[:1])[0]
        self.assertEqual(mapper.valid_mask[int(round(v)), int(round(u))], 0)
        self.assertTrue(CalibrationValidator(profile).check_consistency().passed)

    def test_profile_without_road_plane_or_bev(self):
        bare = new_profile(CAM.intrinsics(), "bare")
        with self.assertRaises(ProfileError):
            CameraGeometry(bare)
        no_bev = CameraGeometry(build_profile(bev=None))
        self.assertEqual(no_bev.image_to_ground([[CAM.cx, 400.0]]).shape, (1, 2))     # image<->ground works without a grid
        with self.assertRaisesRegex(ProfileError, "no BEV"):
            no_bev.ground_to_bev([[1, 0]])
        with self.assertRaises(ProfileError):
            BEVMapper(build_profile(bev=None))


# ---------------------------------------------------------------------------------------------
def blob_centres(row_profile: np.ndarray, threshold: float = 128.0, min_width: int = 2):
    """Sub-pixel centroids of bright runs in a 1-D intensity profile."""
    on = row_profile > threshold
    centres, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if j - i >= min_width:
                w = row_profile[i:j].astype(float)
                centres.append(float((np.arange(i, j) * w).sum() / w.sum()))
            i = j
        else:
            i += 1
    return centres


class BEVMapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapper = BEVMapper(PROFILE)
        cls.straight = cls.mapper.warp(STRAIGHT_IMG)
        cls.curve = cls.mapper.warp(CURVE_IMG)

    # -- output contract ------------------------------------------------------------------
    def test_output_dimensions_dtype_and_masks(self):
        m = self.mapper
        self.assertEqual(self.straight.shape, (500, 350, 3))
        self.assertEqual(self.straight.dtype, np.uint8)
        self.assertEqual(m.valid_mask.shape, (500, 350))
        self.assertEqual(set(np.unique(m.valid_mask)), {0, 255})
        self.assertTrue(((m.support_mask > 0) <= (m.valid_mask > 0)).all())          # supported implies observed
        self.assertEqual(m.shape, (500, 350))

    def test_unobserved_pixels_hold_the_border_value_and_are_masked(self):
        invalid = self.mapper.valid_mask == 0
        self.assertTrue(invalid.any())
        self.assertTrue((self.straight[invalid] == 0).all())
        custom = BEVMapper(PROFILE.with_bev(BEVConfig.from_extents(0.3, 2.3, -0.7, 0.7, MPP, border_value=17)))
        out = custom.warp(STRAIGHT_IMG)
        self.assertTrue((out[custom.valid_mask == 0] == 17).all())
        np.testing.assert_array_equal(custom.valid_mask, self.mapper.valid_mask)      # mask independent of the fill value

    def test_masks_match_the_geometry(self):
        # a BEV pixel is observed iff its ground point projects inside the source image in front of the camera
        m = self.mapper
        vs, us = np.mgrid[0:500:7, 0:350:7]
        uv = np.column_stack([us.ravel(), vs.ravel()]).astype(float)
        px = GEO.bev_to_image(uv)
        expected = GEO.image_contains(px) & (np.isfinite(px).all(axis=1))
        got = m.valid_mask[vs.ravel(), us.ravel()] > 0
        # allow disagreement only within a pixel of the image border
        border = (np.abs(px[:, 0] - 639.0) < 1.0) | (np.abs(px[:, 0]) < 1.0) | (np.abs(px[:, 1] - 479.0) < 1.0) | (np.abs(px[:, 1]) < 1.0)
        self.assertTrue(((expected == got) | border).all())
        # support mask is the calibrated hull grown by the margin, intersected with the observed area
        ground = BEV.bev_to_ground(uv)
        rect = (ground[:, 0] >= 0.5 - 0.1) & (ground[:, 0] <= 1.6 + 0.1) & (ground[:, 1] >= -0.35 - 0.1) & (ground[:, 1] <= 0.35 + 0.1)
        sup = m.support_mask[vs.ravel(), us.ravel()] > 0
        edge = (np.abs(ground[:, 0] - 0.4) < 0.01) | (np.abs(ground[:, 0] - 1.7) < 0.01) | (np.abs(np.abs(ground[:, 1]) - 0.45) < 0.01)
        self.assertTrue(((sup == (rect & got)) | edge).all())

    # -- geometric accuracy ---------------------------------------------------------------
    def lane_centres(self, bev, x_m):
        row = int(round(ORIGIN_V - x_m / MPP))
        return [(c, BEV.bev_to_ground([[c, row]])[0, 1]) for c in blob_centres(bev[row - 1:row + 2, :, :].mean(axis=(0, 2)))]

    def test_straight_lane_markings_land_at_their_true_ground_positions(self):
        for x in (0.7, 1.0, 1.3):
            centres = self.lane_centres(self.straight, x)
            ys = sorted(y for _, y in centres)
            expected = [-0.25, 0.25] if len(ys) == 2 else [-0.25, 0.0, 0.25]
            self.assertEqual(len(ys), len(expected), f"x={x}: {ys}")
            for got, want in zip(ys, expected):
                self.assertAlmostEqual(got, want, delta=0.003, msg=f"x={x}")             # measured: well below 1 px (4 mm)

    def test_metric_scale_lane_width(self):
        centres = [c for c, _ in self.lane_centres(self.straight, 1.0)]
        edges = [centres[0], centres[-1]]
        self.assertAlmostEqual((edges[1] - edges[0]) * MPP, 0.50, delta=0.004)            # lane width is the SYNTHETIC 0.50 m

    def test_left_right_orientation(self):
        centres = self.lane_centres(self.straight, 1.0)
        left_edge = max(centres, key=lambda t: t[1])                                      # y = +0.25 is the vehicle's left edge
        right_edge = min(centres, key=lambda t: t[1])
        self.assertLess(left_edge[0], right_edge[0])                                      # ... and sits further left in the image
        self.assertAlmostEqual(left_edge[1], 0.25, delta=0.003)

    def test_straight_lines_are_straight_and_parallel_in_the_bev(self):
        lo, hi = int(round(ORIGIN_V - 1.5 / MPP)), int(round(ORIGIN_V - 0.6 / MPP))
        for y in (0.25, -0.25):
            u0 = ORIGIN_U - y / MPP
            ys, xs = [], []
            for r in range(lo, hi, 3):
                row = self.straight[r, int(u0) - 6:int(u0) + 7].astype(float).mean(axis=1)
                xs.append(int(u0) - 6 + (np.arange(13) * row).sum() / row.sum()); ys.append(r)
            slope = np.polyfit(ys, xs, 1)[0]
            self.assertLess(abs(np.degrees(np.arctan(slope))), 0.15, msg=f"line y={y}")

    def test_curved_markings_have_the_true_radius(self):
        v_idx, u_idx = np.nonzero((self.curve.mean(axis=2) > 140) & (self.mapper.support_mask > 0))
        g = BEV.bev_to_ground(np.column_stack([u_idx, v_idx]).astype(float))
        sel = g[:, 0] > 1.3                                                               # beyond the straight lead-in
        g = g[sel]
        r = np.hypot(g[:, 0] - 1.2, g[:, 1] - 1.0)
        for radius in (0.75, 1.25):
            on_arc = np.abs(r - radius) < 0.05
            self.assertGreater(on_arc.sum(), 100, f"radius {radius}")
            self.assertAlmostEqual(float(r[on_arc].mean()), radius, delta=0.003)
            # a 2 cm wide marking alone spreads the radius by 0.02/sqrt(12) = 5.8 mm, so allow 30 % on top of that
            self.assertLess(float(r[on_arc].std()), 1.3 * 0.02 / np.sqrt(12))

    def test_curve_bends_toward_the_vehicle_left(self):
        row_far = int(round(ORIGIN_V - 1.55 / MPP))
        row_near = int(round(ORIGIN_V - 1.0 / MPP))
        far = min(blob_centres(self.curve[row_far, :, :].mean(axis=1)))
        near = min(blob_centres(self.curve[row_near, :, :].mean(axis=1)))
        self.assertLess(far, near - 10)                                                   # the left lane edge moves left (smaller u) ahead

    # -- independent cross-checks ---------------------------------------------------------
    def test_fused_remap_agrees_with_undistort_then_warpperspective(self):
        und = self.mapper.undistorter().undistort(STRAIGHT_IMG)
        G = BEV.ground_to_bev_matrix()
        M = PROFILE.road_plane.homography @ np.linalg.inv(G)                              # BEV pixel -> undistorted pixel
        two_step = cv2.warpPerspective(und, M, (350, 500), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        sel = self.mapper.valid_mask > 0
        sel = cv2.erode(sel.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        diff = np.abs(self.straight.astype(int) - two_step.astype(int))[sel]
        self.assertLess(diff.mean(), 3.0)                                                 # measured below; difference is double interpolation
        self.assertGreater(np.mean(diff <= 20), 0.97)

    def test_interpolation_modes_and_map_precision_agree(self):
        base = self.straight.astype(int)
        for mode in ("nearest", "cubic"):
            out = BEVMapper(PROFILE.with_bev(BEVConfig.from_extents(0.3, 2.3, -0.7, 0.7, MPP, interpolation=mode))).warp(STRAIGHT_IMG)
            self.assertLess(np.abs(out.astype(int) - base).mean(), 8, mode)
        fixed = BEVMapper(PROFILE.with_bev(BEVConfig.from_extents(0.3, 2.3, -0.7, 0.7, MPP, fixed_point_maps=True)))
        self.assertIsNotNone(fixed._m2)                                                   # int16 maps carry the interpolation table
        self.assertEqual(fixed._m1.dtype, np.int16)
        self.assertEqual(self.mapper._m1.dtype, np.float32)                               # default: float32 maps
        self.assertLess(np.abs(fixed.warp(STRAIGHT_IMG).astype(int) - base).mean(), 1.0)

    def test_metric_scale_follows_meters_per_pixel_and_origin(self):
        coarse = BEVMapper(PROFILE.with_bev(BEVConfig.from_extents(0.3, 2.3, -0.7, 0.7, 2 * MPP)))
        self.assertEqual(coarse.shape, (250, 175))
        out = coarse.warp(STRAIGHT_IMG)
        row = int(round(coarse.bev.origin_v_px - 1.0 / coarse.bev.meters_per_pixel))
        c = blob_centres(out[row, :, :].mean(axis=1))
        self.assertAlmostEqual((c[-1] - c[0]) * coarse.bev.meters_per_pixel, 0.5, delta=0.008)
        shifted = BEVMapper(PROFILE.with_bev(BEVConfig(350, 500, MPP, ORIGIN_U + 40, ORIGIN_V + 100)))        # origin moved 40 px right, 100 px down
        out2 = shifted.warp(STRAIGHT_IMG)
        row2 = int(round(ORIGIN_V + 100 - 1.0 / MPP))
        shifted_centres = blob_centres(out2[row2, :, :].mean(axis=1))
        np.testing.assert_allclose(shifted_centres[-1] - shifted_centres[0], blob_centres(self.straight[int(round(ORIGIN_V - 1.0 / MPP)), :, :].mean(axis=1))[-1]
                                   - blob_centres(self.straight[int(round(ORIGIN_V - 1.0 / MPP)), :, :].mean(axis=1))[0], atol=1.0)

    # -- region handling ------------------------------------------------------------------
    def test_region_behind_the_camera_is_rejected(self):
        with self.assertRaisesRegex(InvalidBEVRegionError, "no pixel"):
            BEVMapper(PROFILE.with_bev(BEVConfig.from_extents(-2.0, -0.5, -0.5, 0.5, 0.01)))

    def test_partially_observed_region_reports_coverage_and_warnings(self):
        info = self.mapper.info
        self.assertTrue(0.5 < info.valid_fraction < 1.0)
        self.assertTrue(0 < info.support_fraction < info.valid_fraction)
        self.assertGreater(info.footprint_px_max, info.footprint_px_median)
        self.assertTrue(any("anti-aliasing" in w for w in info.warnings))                 # at 4 mm/px the near field is undersampled
        # one BEV pixel covers MORE source pixels on a coarser grid, so a finer grid reduces the problem
        coarse = BEVMapper(PROFILE.with_bev(BEVConfig.from_extents(0.3, 2.3, -0.7, 0.7, 0.01))).info
        fine = BEVMapper(PROFILE.with_bev(BEVConfig.from_extents(0.3, 2.3, -0.7, 0.7, 0.002))).info
        self.assertGreater(coarse.fraction_undersampled, info.fraction_undersampled)
        self.assertLess(fine.fraction_undersampled, info.fraction_undersampled)
        self.assertGreater(coarse.footprint_px_median, info.footprint_px_median)

    def test_region_outside_the_calibrated_hull_warns(self):
        far = BEVMapper(PROFILE.with_bev(BEVConfig.from_extents(2.2, 3.2, -0.3, 0.3, 0.01)))
        self.assertEqual(int((far.support_mask > 0).sum()), 0)
        self.assertTrue(any("calibrated region" in w for w in far.info.warnings))

    # -- input handling -------------------------------------------------------------------
    def test_input_not_modified_even_if_read_only(self):
        img = STRAIGHT_IMG.copy(); img.flags.writeable = False
        before = img.copy()
        out = self.mapper.warp(img)
        np.testing.assert_array_equal(img, before)
        self.assertTrue(out.flags.writeable)
        self.assertFalse(np.shares_memory(out, img))

    def test_grayscale_input(self):
        gray = cv2.cvtColor(STRAIGHT_IMG, cv2.COLOR_BGR2GRAY)
        out = self.mapper.warp(gray)
        self.assertEqual(out.shape, (500, 350))

    def test_bad_inputs(self):
        for bad in (np.zeros((480, 640, 3), np.float32), np.zeros((480, 640, 4), np.uint8), np.zeros((0, 640, 3), np.uint8),
                    [[0]], None):
            with self.subTest(type=type(bad).__name__), self.assertRaises(InvalidFrameError):
                self.mapper.warp(bad)
        with self.assertRaises(ResolutionMismatchError):
            self.mapper.warp(np.zeros((240, 320, 3), np.uint8))

    def test_destination_buffer_validation_and_reuse(self):
        dst = np.empty((500, 350, 3), np.uint8)
        out = self.mapper.warp(STRAIGHT_IMG, dst=dst)
        self.assertIs(out, dst)
        np.testing.assert_array_equal(dst, self.straight)
        for bad in (np.empty((10, 10, 3), np.uint8), np.empty((500, 350, 3), np.float32)):
            with self.assertRaises(InvalidFrameError):
                self.mapper.warp(STRAIGHT_IMG, dst=bad)
        ro = np.empty((500, 350, 3), np.uint8); ro.flags.writeable = False
        with self.assertRaises(InvalidFrameError):
            self.mapper.warp(STRAIGHT_IMG, dst=ro)

    def test_maps_are_shared_between_equal_profiles_and_read_only(self):
        mapper_mod.clear_map_cache()
        a, b = BEVMapper(PROFILE), BEVMapper(PROFILE)
        self.assertIs(a._m1, b._m1)
        for arr in (a._m1, a.valid_mask, a.support_mask):
            with self.assertRaises(ValueError):
                arr[0, 0] = 1
        other = BEVMapper(PROFILE.with_bev(BEVConfig.from_extents(0.3, 2.3, -0.7, 0.7, 0.005)))
        self.assertIsNot(other._m1, a._m1)

    def test_concurrent_warps_are_consistent(self):
        results = []
        def work():
            results.append(self.mapper.warp(STRAIGHT_IMG))
        threads = [threading.Thread(target=work) for _ in range(6)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(len(results), 6)
        for r in results:
            np.testing.assert_array_equal(r, self.straight)

    def test_fisheye_camera_produces_a_correct_bev(self):
        pose = SyntheticPose(0.25, 35.0)
        prof = build_profile(FISH, pose)
        m = BEVMapper(prof)
        out = m.warp(make_road_scene("straight").render(FISH, pose))
        self.assertGreater(m.info.valid_fraction, 0.3)
        row = int(round(ORIGIN_V - 1.0 / MPP))
        c = blob_centres(out[row - 1:row + 2, :, :].mean(axis=(0, 2)))
        self.assertGreaterEqual(len(c), 2)
        self.assertAlmostEqual((c[-1] - c[0]) * MPP, 0.5, delta=0.006)


if __name__ == "__main__":
    unittest.main()
