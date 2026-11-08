"""Error-sensitive evidence score and conservative adversarial geometry tests."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np

from . import LaneConfig, LaneDetector
from .adversarial import arc_scene
from .fitting import fit_boundary
from .synthetic import bev_fixture


class ResidualQualityTests(unittest.TestCase):
    def test_default_hough_is_off_and_optional_votes_preserve_geometry(self):
        image,_=arc_scene(700,100)
        frame=bev_fixture(image)
        self.assertFalse(LaneConfig().use_hough)
        with patch('cv2.HoughLinesP',side_effect=AssertionError('default Hough must be off')):
            off=LaneDetector().process(frame)
        on=LaneDetector(LaneConfig(use_hough=True)).process(frame)
        self.assertTrue(off.valid)
        self.assertTrue(on.valid)
        self.assertEqual(off.usable_support_v_px,on.usable_support_v_px)
        np.testing.assert_array_equal(off.selected_pair.centreline.coefficients,on.selected_pair.centreline.coefficients)
        self.assertTrue(any(b.diagnostics.hough_support_fraction>0 for b in on.boundaries))

    def test_residual_growth_reduces_confidence_at_identical_support(self):
        rows=np.arange(8,234,3,dtype=float)
        rng=np.random.default_rng(123)
        noise=rng.uniform(-1,1,len(rows))
        fits=[fit_boundary(np.column_stack((70+a*noise,rows)),LaneConfig(),240,0)
              for a in (0,.5,3.8)]
        self.assertTrue(all(b is not None for b in fits))
        self.assertGreater(fits[0].confidence,fits[1].confidence)
        self.assertGreater(fits[1].confidence,fits[2].confidence)
        self.assertLess(fits[2].diagnostics.residual_quality_factor,1)
        self.assertIn('fit_error_reduces_confidence',fits[2].reason_codes)
        self.assertGreater(fits[2].diagnostics.coherent_bias_max_px,1)

    def test_full_curve_error_is_reported_and_hairpin_is_not_forced_valid(self):
        refused=0
        for radius in (120,140,160):
            image,circle=arc_scene(radius,60)
            r=LaneDetector().process(bev_fixture(image))
            if r.valid:
                points=r.selected_pair.centreline.samples(100)
                error=np.max(np.abs(np.linalg.norm(points-np.asarray(circle[:2]),axis=1)-radius))
                self.assertLess(error,2)
                lo,hi=r.usable_support_v_px
                self.assertTrue(np.isnan(r.evaluate_supported_centreline([lo-1,hi+1])).all())
            else:
                refused+=1
                self.assertIsNone(r.selected_pair)
                self.assertIn('coherent_model_misfit',r.reason_codes)
        self.assertEqual(refused,2)
        image,_=arc_scene(700,100)
        r=LaneDetector().process(bev_fixture(image))
        self.assertTrue(r.valid,r.reason_codes)
        for b in r.boundaries:
            self.assertGreaterEqual(b.diagnostics.all_observation_p95_px,0)
            self.assertGreaterEqual(b.diagnostics.coherent_bias_max_px,0)
            self.assertGreater(b.diagnostics.residual_quality_factor,0)
            self.assertLessEqual(b.diagnostics.residual_quality_factor,1)

    def test_broad_colored_materials_and_horizontal_glare_are_rejected(self):
        for color in ((140,230,240),(208,253,255),(0,215,255)):
            r=LaneDetector().process(bev_fixture(np.full((240,320,3),color,np.uint8)))
            self.assertFalse(r.valid)
            self.assertIsNone(r.selected_pair)
        image=np.full((240,320,3),35,np.uint8)
        for v in (70,200):
            cv2.line(image,(5,v),(314,v),(255,255,255),5)
        r=LaneDetector().process(bev_fixture(image))
        self.assertFalse(r.valid)
        self.assertIsNone(r.selected_pair)


if __name__=='__main__':
    unittest.main()
