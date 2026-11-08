import unittest

import cv2
import numpy as np

from . import LaneConfig, LaneDetector
from .availability_cases import transition_case
from .extraction import extract
from .fitting import fit_boundary
from .preprocessing import Preprocessor
from .synthetic import bev_fixture, scene


def staggered_dashes():
    image=np.full((240,320,3),35,np.uint8)
    for u,offset in ((70,6),(200,26)):
        for start in range(offset,234,40):
            cv2.line(image,(u,start),(u,min(start+10,239)),(255,255,255),5)
    return image


class EvidencePolicyTests(unittest.TestCase):
    def test_limited_overlay_never_samples_outside_fractional_domain(self):
        import warnings
        from unittest.mock import patch
        from .models import Polynomial
        from .visualization import _curve
        image=np.zeros((240,320,3),np.uint8)
        p=Polynomial((0,0,100),120.25,230.0,allow_extrapolation=False)
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            with patch.object(p.__class__,'evaluate',autospec=True,side_effect=Polynomial.evaluate) as evaluated:
                _curve(image,p,(0,255,0),((0,239),))
                rows=evaluated.call_args.args[1]
                self.assertTrue(np.all((rows>=p.v_min)&(rows<=p.v_max)))
        self.assertGreater(np.count_nonzero(image),0)

    def test_yellow_floor_and_tapered_floor_bands_are_refused_but_yellow_paint_detected(self):
        yellow=(0,230,230)
        result=LaneDetector().process(bev_fixture(np.full((240,320,3),yellow,np.uint8)))
        self.assertFalse(result.valid)
        self.assertIn('saturated_marking_evidence',result.reason_codes)
        image=np.full((240,320,3),35,np.uint8)
        for v in range(240):
            half=int(round(2+7*v/239))
            for u in (70,200):
                image[v,u-half:u+half+1]=yellow
        result=LaneDetector().process(bev_fixture(image))
        self.assertFalse(result.valid)
        self.assertIn('marking_width_inconsistent',result.reason_codes)
        image[:]=35
        for u in (70,200):
            cv2.line(image,(u,6),(u,233),yellow,5)
        result=LaneDetector().process(bev_fixture(image))
        self.assertTrue(result.valid,result.reason_codes)
        self.assertFalse(result.metric_validated)
        self.assertIsNone(result.selected_pair.width_mean_m)

    def test_tapered_glare_streaks_are_not_accepted_as_consistent_paint(self):
        for reverse in (False,True):
            image=np.full((240,320,3),35,np.uint8)
            for v in range(240):
                half=int(round(2+7*(v/239 if reverse else 1-v/239)))
                for u in (70,200):
                    image[v,u-half:u+half+1]=255
            detector=LaneDetector()
            r=detector.process(bev_fixture(image),debug=True)
            self.assertFalse(r.valid)
            self.assertIn('marking_width_inconsistent',r.reason_codes)
            self.assertTrue(any(not b.marking_evidence_valid for b in r.boundaries))
            self.assertFalse(np.any(detector.last_debug.accepted_mask))

    def test_zero_shared_paint_is_explicit_and_configurable(self):
        f=bev_fixture(staggered_dashes())
        r=LaneDetector().process(f)
        self.assertFalse(r.valid)
        self.assertIsNone(r.selected_pair)
        self.assertIn('no_shared_observed_support',r.reason_codes)
        self.assertTrue(r.compatible_pairs)
        self.assertEqual(r.compatible_pairs[0].shared_observed_fraction,0)
        allowed=LaneDetector(LaneConfig(require_shared_observation=False)).process(f)
        self.assertTrue(allowed.valid,allowed.reason_codes)
        self.assertIn('no_shared_observed_support',allowed.reason_codes)
        self.assertLess(allowed.confidence,LaneDetector().process(bev_fixture(scene(((0,0,70),(0,0,200))))).confidence)

    def test_long_occlusion_is_not_called_a_spatial_fork(self):
        image=scene(((0,0,70),(0,0,200)))
        image[80:160]=35
        r=LaneDetector().process(bev_fixture(image))
        self.assertFalse(r.valid)
        self.assertIn('disjoint_lane_support',r.reason_codes)
        self.assertNotIn('ambiguous_geometry',r.reason_codes)
        self.assertNotIn('multiple_compatible_pairs',r.reason_codes)

    def test_budget_exhaustion_is_not_called_a_spatial_fork(self):
        r=LaneDetector(LaneConfig(max_association_comparisons=1)).process(bev_fixture(scene()))
        self.assertIn('processing_budget_exceeded',r.reason_codes)
        self.assertNotIn('ambiguous_geometry',r.reason_codes)

    def test_fit_reports_all_observations_and_coherent_misfit(self):
        c=LaneConfig()
        f=bev_fixture(transition_case(300,150))
        e=extract(Preprocessor(c).process(f),c)
        fits=[fit_boundary(p,c,240,i) for i,p in enumerate(e.tracks)]
        self.assertTrue(any(b is not None and b.model_misfit_intervals for b in fits))
        self.assertTrue(any(b is not None and b.diagnostics.all_observation_p95_px>1.5 for b in fits))
        r=LaneDetector().process(f)
        self.assertTrue(r.valid)
        self.assertTrue(r.limited_support)
        self.assertTrue(any(b.diagnostics.all_observation_rms_px>b.diagnostics.residual_rms_px+1 for b in r.boundaries))
