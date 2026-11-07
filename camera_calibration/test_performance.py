"""Performance regression tests (SYNTHETIC frames, this machine only).

Bounds are deliberately loose: they exist to catch structural regressions (maps rebuilt per frame, unbounded memory,
first-frame stalls), not to certify speed. Actual numbers come from ``python -m camera_calibration.benchmark``;
nothing here says anything about a Raspberry Pi 5.
"""
import time
import unittest
from unittest.mock import patch

from camera_calibration import mapper as mapper_mod
from camera_calibration.benchmark import build_case, end_to_end_age, memory_growth, time_processor
from camera_calibration.mapper import BEVMapper
from camera_calibration.pipeline import BEVProcessor, ProcessorConfig

PROFILE, FRAME = build_case(640, 480)


class StructuralPerformanceTests(unittest.TestCase):
    def test_maps_are_built_once_not_per_frame(self):
        proc = BEVProcessor(PROFILE, ProcessorConfig(max_frame_age_s=None))
        from camera_calibration.benchmark import _packet
        with patch.object(mapper_mod, "_compute_maps", side_effect=AssertionError("maps rebuilt while processing")) as spy:
            for i in range(60):
                proc.process(_packet(FRAME, i + 1))
            self.assertEqual(spy.call_count, 0)

    def test_setup_cost_is_paid_up_front_and_dwarfs_a_warm_frame(self):
        r = time_processor(PROFILE, FRAME, frames=60)
        self.assertGreater(r["setup_ms"], 10 * r["mean_ms"])           # measured ~230 ms vs ~0.15 ms
        self.assertLess(r["first_frame_ms"], 0.5 * r["setup_ms"])       # the first frame does NOT pay for the map build
        self.assertLess(r["mean_ms"], 50.0)                             # pathological-regression guard only
        self.assertLess(r["max_ms"], 500.0)

    def test_equal_profiles_share_built_maps(self):
        mapper_mod.clear_map_cache()
        t0 = time.perf_counter(); BEVMapper(PROFILE); cold = time.perf_counter() - t0
        t0 = time.perf_counter(); BEVMapper(PROFILE); warm = time.perf_counter() - t0
        self.assertLess(warm, cold / 5)

    def test_processing_cost_is_output_driven_not_source_driven(self):
        small, big = build_case(640, 480), build_case(1920, 1080)
        a = time_processor(*small, frames=60)["mean_ms"]
        b = time_processor(*big, frames=60)["mean_ms"]
        self.assertLess(b, 8 * a + 1.0)                                  # measured ~1.2x; a 9x-larger source must not cost 9x

    def test_memory_does_not_grow_with_frames(self):
        grown = memory_growth(PROFILE, FRAME, frames=200)
        one_bev = PROFILE.bev.width_px * PROFILE.bev.height_px * 3
        self.assertLess(grown, 4 * one_bev)                              # measured: exactly one retained BEV

    def test_end_to_end_frame_age_through_the_real_m01_manager(self):
        e = end_to_end_age(PROFILE, FRAME, seconds=1.0, source_fps=60.0)
        self.assertGreater(e["frames"], 20)
        self.assertLess(e["p95_ms"], 100.0)                              # measured ~1 ms: processing + wake-up, software only
        self.assertLessEqual(e["skipped"], 0.2 * e["frames"])            # the worker keeps up with the source


if __name__ == "__main__":
    unittest.main()
