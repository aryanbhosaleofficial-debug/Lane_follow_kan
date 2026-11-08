"""Test-first regressions for review R-09 and R-08."""
import unittest

import numpy as np

from . import LaneConfig, LaneDetector
from .availability_cases import availability_frame, clutter_case, transition_case
from .synthetic import bev_fixture


class ClutterAvailabilityTests(unittest.TestCase):
    def test_curve_dash_and_competing_candidates_under_clutter(self):
        for scale in (1, 2):
            for geometry in ("curve", "dashed", "competing"):
                for kind, count in (("fragments", 60), ("specks", 200)):
                    valid = 0
                    for seed in range(8):
                        frame = availability_frame(clutter_case(kind, count, seed, scale, geometry), scale)
                        cfg = LaneConfig.for_bev(frame.bev, (.36, .68), max_frame_age_s=None)
                        r = LaneDetector(cfg).process(frame)
                        if r.valid:
                            self.assertNotEqual(geometry, "competing")
                            p = r.selected_pair.centreline
                            rows = np.linspace(p.v_min, p.v_max, 100) / scale
                            truth = 135 + (.001 * (rows - 120) ** 2 if geometry == "curve" else 0)
                            self.assertLess(np.max(np.abs(p.evaluate(rows * scale) / scale - truth)), 2.5)
                            valid += 1
                    with self.subTest(scale=scale, geometry=geometry, kind=kind):
                        self.assertEqual(valid, 0) if geometry == "competing" else self.assertGreaterEqual(valid, 7)

    def test_frame_processing_does_not_accumulate_tracks(self):
        import gc
        import tracemalloc
        from dataclasses import replace
        frame = availability_frame(clutter_case("specks", 200, 0))
        cfg = LaneConfig(max_frame_age_s=None)
        detector = LaneDetector(cfg)
        tracemalloc.start()
        try:
            for i in range(10):
                detector.process(replace(frame, source_frame_id=i + 1, bev_sequence=i + 1))
            gc.collect()
            first = tracemalloc.get_traced_memory()[0]
            for i in range(10, 60):
                result = detector.process(replace(frame, source_frame_id=i + 1, bev_sequence=i + 1))
                d = result.extraction_diagnostics
                self.assertLessEqual(d.peak_active_tracks, cfg.max_tracks)
                self.assertLessEqual(d.association_comparisons, cfg.max_association_comparisons)
                self.assertLessEqual(d.peak_stored_points, (cfg.max_tracks + cfg.max_fit_candidates + 1) * 81)
            gc.collect()
            live, peak = tracemalloc.get_traced_memory()
            self.assertLess(live - first, frame.image.nbytes * 2 + 256_000)
        finally:
            tracemalloc.stop()

    def test_track_retirement_reuses_active_capacity(self):
        from .extraction import extract
        from .preprocessing import Preprocessor
        frame = availability_frame(clutter_case("specks", 200, 0))
        cfg = LaneConfig(max_frame_age_s=None)
        output = extract(Preprocessor(cfg).process(frame), cfg)
        d = output.diagnostics
        self.assertGreater(d.tracks_created, cfg.max_tracks)
        self.assertGreater(d.retired_tracks, 48)
        self.assertLessEqual(d.peak_active_tracks, cfg.max_tracks)
        self.assertLessEqual(d.peak_completed_candidates, cfg.max_fit_candidates)
        self.assertFalse(output.overloaded)
        self.assertGreaterEqual(len(output.tracks), 2)

    def test_eight_seed_fragment_recall(self):
        for scale in (1, 2):
            correct, wrong = 0, 0
            for seed in range(8):
                frame = availability_frame(clutter_case("fragments", 60, seed, scale), scale)
                cfg = LaneConfig.for_bev(frame.bev, (.36, .68), max_frame_age_s=None)
                r = LaneDetector(cfg).process(frame)
                if r.valid:
                    error = abs(float(r.selected_pair.centreline.evaluate(180 * scale)) / scale - 135)
                    correct += error < 2.5
                    wrong += error >= 2.5
            with self.subTest(scale=scale):
                # The preserved earlier implementation achieves 8/8 on this
                # independent 320px protocol; recover it, not merely 5/8.
                self.assertEqual(correct, 8)
                self.assertEqual(wrong, 0)

    def test_eight_seed_speck_recall(self):
        for scale in (1, 2):
            good = 0
            for seed in range(8):
                frame = availability_frame(clutter_case("specks", 200, seed, scale), scale)
                cfg = LaneConfig.for_bev(frame.bev, (.36, .68), max_frame_age_s=None)
                r = LaneDetector(cfg).process(frame)
                if r.valid:
                    self.assertLess(abs(float(r.selected_pair.centreline.evaluate(180 * scale)) / scale - 135), 2.5)
                    good += 1
            with self.subTest(scale=scale):
                self.assertEqual(good, 8)


class TransitionAvailabilityTests(unittest.TestCase):
    def test_ground_truth_transition_matrix_requires_available_pairs(self):
        from .transition_evaluation import run
        report = run()
        self.assertEqual(report['failed'],0)
        self.assertEqual(report['required'],60)
        self.assertEqual(report['required_recovered'],60)
        self.assertGreaterEqual(report['valid'],60)
        self.assertGreater(report['limited'],0)

    def test_partially_occluded_boundary_keeps_scoped_gap(self):
        image=transition_case(220,130)
        image[175:195,194:207]=35
        r=LaneDetector().process(bev_fixture(image))
        self.assertTrue(r.valid,r.reason_codes)
        self.assertTrue(r.limited_support)
        self.assertFalse(r.selected_pair.inferred)
        self.assertTrue(r.selected_pair.unsupported_intervals)
        self.assertLess(r.selected_pair.observed_fraction,.98)

    def test_only_rows_200_to_240_of_arc_are_visible(self):
        f = availability_frame(transition_case(220,120,arc_visible_px=20,scale=2),2)
        r = LaneDetector(LaneConfig.for_bev(f.bev,(.36,.68),max_frame_age_s=None)).process(f)
        self.assertTrue(r.valid,r.reason_codes)
        self.assertGreater(r.selected_pair.centreline.v_max-r.selected_pair.centreline.v_min,130)

    def test_missing_boundary_and_forks_remain_uncertain(self):
        import cv2
        for direction in (1,-1):
            r = LaneDetector().process(bev_fixture(transition_case(220,130,direction=direction,missing=True)))
            self.assertFalse(r.valid)
            self.assertIsNone(r.selected_pair)
        image = transition_case(220,130)
        cv2.line(image,(200,205),(290,50),(255,255,255),5)
        r=LaneDetector().process(bev_fixture(image))
        self.assertFalse(r.valid)
        self.assertIsNone(r.selected_pair)

    def test_supported_near_field_survives_far_curve(self):
        for direction in (1, -1):
            image = transition_case(220, 130, direction=direction)
            r = LaneDetector().process(bev_fixture(image))
            self.assertTrue(r.valid, r.reason_codes)
            self.assertTrue(r.selected_pair.limited_support)
            self.assertTrue(r.selected_pair.unsupported_far_intervals)
            p = r.selected_pair.centreline
            self.assertGreaterEqual(p.v_min, 125)
            self.assertFalse(p.allow_extrapolation)
            self.assertTrue(np.isnan(p.evaluate([p.v_min - 1, p.v_max + 1])).all())
            self.assertEqual(r.usable_support_v_px, (p.v_min, p.v_max))
            rows = np.linspace(max(p.v_min, 130), p.v_max, 50)
            truth = 135 if direction == 1 else 319 - 135
            self.assertLess(np.max(np.abs(p.evaluate(rows) - truth)), 1.5)
