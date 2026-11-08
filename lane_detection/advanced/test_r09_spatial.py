"""Original review protocol and bounded spatial association regressions."""
from dataclasses import replace
import unittest

import cv2
import numpy as np

from . import LaneConfig, LaneDetector
from .availability_cases import availability_frame
from .extraction import extract
from .preprocessing import Preprocessor
from .r09_cases import review_case


class SpatialBudgetTests(unittest.TestCase):
    def test_exact_reviewer_60_fragment_protocol_and_conservative_alternatives(self):
        for scale, expected in ((1, 6), (2, 7)):
            correct = 0
            for seed in range(8):
                frame = availability_frame(review_case('fragments',60,seed,scale),scale)
                cfg = LaneConfig.for_bev(frame.bev,(.18,1.04),max_frame_age_s=None)
                r = LaneDetector(cfg).process(frame)
                self.assertIsNone(r.extraction_diagnostics.budget_reason)
                self.assertLess(r.extraction_diagnostics.association_comparisons,700)
                if r.valid:
                    by_id = {b.candidate_id:b for b in r.boundaries}
                    for identifier, truth in ((r.selected_pair.left_id,70),(r.selected_pair.right_id,200)):
                        p=by_id[identifier].polynomial
                        rows=np.linspace(p.v_min,p.v_max,100)
                        self.assertLess(np.max(np.abs(p.evaluate(rows)/scale-truth)),2)
                    correct += 1
                else:
                    self.assertIsNone(r.selected_pair)
                    self.assertTrue({'competing_close_markings','branch_geometry'} & set(r.reason_codes))
                    self.assertGreater(r.extraction_diagnostics.peak_completed_candidates,2)
                    if 'competing_close_markings' in r.reason_codes:
                        self.assertGreater(len(r.boundaries),2)
            # Extra fragment chains are competing evidence under the existing
            # dashed/companion contract. Never remove that veto to force recall.
            self.assertEqual(correct,expected)

    def test_requested_clutter_levels_never_spend_budget_on_distant_runs(self):
        for scale in (1,2):
            for kind,counts in (('fragments',(40,60,80,100)),('specks',(100,200,300))):
                for count in counts:
                    for seed in range(8):
                        frame=availability_frame(review_case(kind,count,seed,scale),scale)
                        cfg=LaneConfig.for_bev(frame.bev,(.18,1.04),max_frame_age_s=None)
                        out=extract(Preprocessor(cfg).process(frame),cfg)
                        self.assertFalse(out.overloaded,(scale,kind,count,seed,out.diagnostics))
                        self.assertLessEqual(out.diagnostics.association_comparisons,cfg.max_association_comparisons)
                        self.assertLessEqual(out.diagnostics.peak_active_tracks,cfg.max_tracks)

    def test_exact_budget_boundary_is_inclusive_and_exhaustion_refuses_pair(self):
        frame=availability_frame(review_case('fragments',0,0))
        cfg=LaneConfig(max_frame_age_s=None)
        clean=LaneDetector(cfg).process(frame)
        work=clean.extraction_diagnostics.association_comparisons
        self.assertEqual(work,154)  # one local candidate per edge per subsequent row
        self.assertTrue(LaneDetector(replace(cfg,max_association_comparisons=work)).process(frame).valid)
        r=LaneDetector(replace(cfg,max_association_comparisons=work-1)).process(frame)
        self.assertFalse(r.valid)
        self.assertIsNone(r.selected_pair)
        self.assertEqual(r.extraction_diagnostics.budget_reason,'association_comparison_limit')
        self.assertLessEqual(r.extraction_diagnostics.association_comparisons,work-1)

    def test_candidate_order_and_clutter_retirement_are_deterministic(self):
        frame=availability_frame(review_case('fragments',100,3))
        cfg=LaneConfig(use_hough=False,max_frame_age_s=None)
        pre=Preprocessor(cfg).process(frame)
        a,b=extract(pre,cfg),extract(pre,cfg)
        self.assertEqual(a.diagnostics,b.diagnostics)
        self.assertGreater(a.diagnostics.retired_tracks,0)
        self.assertEqual(len(a.tracks),len(b.tracks))
        for first,second in zip(a.tracks,b.tracks):
            np.testing.assert_array_equal(first,second)

    def test_crowded_local_candidates_still_exhaust_budget(self):
        # Distinct narrow runs inside the same gates must still count individually.
        image=np.full((240,320,3),35,np.uint8)
        for u in range(60,130,5):
            cv2.line(image,(u,5),(u,233),(255,255,255),1)
        cfg=LaneConfig(blur_size=1,max_association_comparisons=100,max_frame_age_s=None)
        r=LaneDetector(cfg).process(availability_frame(image))
        self.assertFalse(r.valid)
        self.assertIsNone(r.selected_pair)
        self.assertEqual(r.extraction_diagnostics.budget_reason,'association_comparison_limit')


if __name__=='__main__':
    unittest.main()
