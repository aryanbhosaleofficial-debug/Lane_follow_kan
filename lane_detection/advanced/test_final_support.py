"""Planner support queries, actual M02 origin and reference-extension boundaries."""
from dataclasses import replace
from unittest.mock import patch
import unittest

import numpy as np

from camera_calibration import BEVProcessor
from camera_calibration.test_bev import PROFILE
from camera_calibration.test_pipeline import packet
from . import LaneConfig, LaneDetector
from .availability_cases import transition_case
from .synthetic import bev_fixture, scene
from . import test_final_interfaces as interfaces
from .test_review_integration import physical_scene


class SupportReferenceTests(unittest.TestCase):
    def test_actual_m02_origin_outside_bev_needs_explicit_reference(self):
        f=BEVProcessor(PROFILE).process(packet(image=physical_scene()[0]))
        self.assertGreater(f.bev.origin_v_px,f.bev.height_px-1)
        absent=LaneDetector().process(f)
        self.assertIsNone(absent.vehicle_reference_bev_px)
        self.assertFalse(absent.lane_reference_valid)
        origin=LaneDetector(LaneConfig(ground_origin_is_vehicle=True)).process(f)
        self.assertFalse(origin.lane_reference_valid)
        self.assertIn('vehicle_reference_outside_bev',origin.reference_reason_codes)
        self.assertIsNone(origin.ego_offset_px)
        self.assertTrue(absent.valid)
        p=absent.selected_pair.centreline
        v=(p.v_min+p.v_max)/2
        u=float(p.evaluate(v))
        explicit=LaneDetector(LaneConfig(vehicle_reference_bev_px=(u,v))).process(f)
        self.assertTrue(explicit.ego_lane_valid,explicit.reference_reason_codes)
        self.assertFalse(explicit.metric_validated)
        self.assertIsNone(explicit.ego_offset_m)

    def test_limited_pair_query_cannot_extrapolate(self):
        r=LaneDetector().process(bev_fixture(transition_case(220,130)))
        self.assertTrue(r.valid)
        self.assertTrue(r.limited_support)
        lo,hi=r.usable_support_v_px
        rows=np.array([lo-1,lo,(lo+hi)/2,hi,hi+1])
        values=r.evaluate_supported_centreline(rows)
        self.assertTrue(np.isnan(values[[0,4]]).all())
        self.assertTrue(np.isfinite(values[1:4]).all())
        self.assertTrue(np.isnan(r.evaluate_supported_centreline(lo-1)))
        self.assertTrue(np.isfinite(r.evaluate_supported_centreline((lo+hi)/2)))
        self.assertIn('limited_near_field',r.reason_codes)
        self.assertEqual(r.to_dict()['usable_support_v_px'],r.usable_support_v_px)

    def test_full_pair_query_masks_domain_gaps_and_invalid_observations(self):
        r=LaneDetector().process(bev_fixture(scene()))
        lo,hi=r.usable_support_v_px
        self.assertTrue(np.isnan(r.evaluate_supported_centreline([lo-1,hi+1])).all())
        p=replace(r.selected_pair,unsupported_intervals=((100.,120.),))
        gap=replace(r,selected_pair=p)
        self.assertTrue(np.isnan(gap.evaluate_supported_centreline([100,110,120])).all())
        self.assertTrue(np.isfinite(gap.evaluate_supported_centreline([99,121])).all())
        for invalid in (replace(r,validity='uncertain'),replace(r,selected_pair=None),replace(r,usable_support_v_px=None)):
            self.assertTrue(np.isnan(invalid.evaluate_supported_centreline([100,120])).all())

    def test_reference_cap_at_below_and_above_six_pixels(self):
        f=bev_fixture(interfaces.FinalInterfaceTests().bottom_paint())
        # Move the fitted endpoint inward while retaining raw paint. Above-cap
        # queries must remain inside the raster, so the BEV bounds check cannot
        # hide a missing numeric extension cap.
        base=LaneDetector(LaneConfig(fit_endpoint_margin_px=6)).process(f)
        hi=base.selected_pair.centreline.v_max
        self.assertLess(hi+6.01,f.bev.height_px-1)
        for delta,expected in ((0,True),(6-1e-6,True),(6,True),(6+1e-6,False)):
            cfg=LaneConfig(vehicle_reference_bev_px=(135,hi+delta),fit_endpoint_margin_px=6)
            r=LaneDetector(cfg).process(f)
            self.assertEqual(r.lane_reference_valid,expected,(hi,delta,r.reference_reason_codes))
            if delta>0 and expected:
                self.assertTrue(r.reference_extrapolated)
                self.assertAlmostEqual(r.reference_extension_px,delta)
                self.assertTrue(np.isnan(r.evaluate_supported_centreline(hi+delta)))
        # Even a permissive raw-evidence check cannot bypass the numeric cap.
        with patch('lane_detection.advanced.preprocessing.Preprocessor.reference_evidence',return_value=True) as evidence:
            r=LaneDetector(LaneConfig(vehicle_reference_bev_px=(135,hi+6.01),fit_endpoint_margin_px=6)).process(f)
            self.assertFalse(r.lane_reference_valid)
            evidence.assert_not_called()

    def test_zero_cap_far_extension_and_outside_grid_cannot_bypass(self):
        f=bev_fixture(interfaces.FinalInterfaceTests().bottom_paint())
        base=LaneDetector().process(f)
        lo,hi=base.usable_support_v_px
        for ref,cap in (((135,hi+.01),0),((135,lo-.01),6),((-1,120),6),((320,120),1000),((135,240),1000)):
            r=LaneDetector(LaneConfig(vehicle_reference_bev_px=ref,max_reference_extension_px=cap)).process(f)
            self.assertFalse(r.lane_reference_valid,(ref,r.reference_reason_codes))
            self.assertFalse(r.reference_extrapolated)
            self.assertIsNone(r.ego_offset_px)


if __name__=='__main__':
    unittest.main()
