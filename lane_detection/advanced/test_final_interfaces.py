from dataclasses import fields, replace
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from camera_calibration import BEVConfig
from . import LaneConfig, LaneDetector
from .cli import main
from .synthetic import bev_fixture, synthetic_profile


class FinalInterfaceTests(unittest.TestCase):
    def bottom_paint(self):
        image=np.full((240,320,3),35,np.uint8)
        for u in (70,200):
            cv2.line(image,(u,0),(u,239),(255,255,255),5)
        return image

    def test_bottom_reference_uses_bounded_visible_evidence(self):
        cfg=LaneConfig(vehicle_reference_bev_px=(135,239))
        r=LaneDetector(cfg).process(bev_fixture(self.bottom_paint()))
        self.assertTrue(r.valid)
        self.assertTrue(r.lane_reference_valid)
        self.assertTrue(r.brackets_origin)
        self.assertTrue(r.reference_extrapolated)
        self.assertLessEqual(r.reference_extension_px,6)
        self.assertIsNone(r.ego_offset_m)

    def test_reference_extension_needs_both_markings_and_support(self):
        image=self.bottom_paint()
        image[236:,194:207]=35
        r=LaneDetector(LaneConfig(vehicle_reference_bev_px=(135,239))).process(bev_fixture(image))
        self.assertFalse(r.lane_reference_valid)
        f=bev_fixture(self.bottom_paint())
        mask=f.support_mask.copy();mask[238:]=0
        r=LaneDetector(LaneConfig(vehicle_reference_bev_px=(135,239))).process(replace(f,support_mask=mask))
        self.assertFalse(r.lane_reference_valid)

    def test_factory_refuses_unreasonable_scale_ratios(self):
        for mpp in (1e-9,1.0,.0005,.032):
            with self.subTest(mpp=mpp),self.assertRaises(ValueError):
                LaneConfig.for_bev(BEVConfig(320,240,mpp,160,239),(.36,.68))
        for mpp in (.016,.004,.001):
            LaneConfig.for_bev(BEVConfig(320,240,mpp,160,239),(.36,.68))

    def test_undecodable_image_is_clean_cli_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);image=root/'bad.png';image.write_text('not an image')
            profile=root/'profile.json';synthetic_profile().save(profile)
            error=io.StringIO()
            with contextlib.redirect_stderr(error),self.assertRaises(SystemExit) as result:
                main(['--source',str(image),'--profile',str(profile),'--headless','--output',str(root/'out')])
            self.assertEqual(result.exception.code,2)
            self.assertIn('cannot decode image',error.getvalue())
            self.assertNotIn('Traceback',error.getvalue())

    def test_original_positional_order_is_preserved(self):
        names=('white_min_value','white_max_saturation','yellow_min_value','yellow_min_saturation',
               'yellow_hue_min','yellow_hue_max','preprocessing','local_contrast','local_window','clahe',
               'blur_size','opening_size','roi_margin_px','canny_low','canny_high','hough_threshold',
               'hough_min_length','hough_max_gap','max_abs_du_dv','segment_duplicate_px','row_step',
               'max_marking_width_px','association_margin_px','max_gap_px','max_tracks','max_row_runs',
               'min_points','min_span_px','min_coverage','max_residual_px','inlier_threshold_px',
               'max_outlier_fraction','quadratic_min_span_px','quadratic_improvement_px','max_curvature_px_inv',
               'min_pair_overlap_px','min_lane_width_px','max_lane_width_px','max_width_variation',
               'min_pair_confidence','lane_width_m','max_curvature_m_inv','nominal_lane_width_px',
               'nominal_lane_width_m','single_boundary_role')
        defaults=LaneConfig()
        values=[getattr(defaults,n) for n in names]
        config=LaneConfig(*values,use_hough=False)
        self.assertEqual(tuple(f.name for f in fields(LaneConfig) if not f.kw_only),names)
        self.assertEqual([getattr(config,n) for n in names],values)
        self.assertFalse(config.use_hough)
