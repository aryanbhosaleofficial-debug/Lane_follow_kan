"""Adversarial semantics and physical-circle operating-range audit (no claims)."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from . import LaneConfig, LaneDetector
from .adversarial import arc_scene
from .availability_evaluation import engine_at
from .synthetic import bev_fixture


def run(directory=None):
    engine=engine_at(directory)
    semantics=[]
    for name,color in (('white',(255,255,255)),('yellow',(0,230,230)),
                       ('khaki',(140,230,240)),('cream',(208,253,255)),('gold',(0,215,255))):
        for material in ('paint_control','wall_streak','glare_streak','floor'):
            image=np.full((240,320,3),35,np.uint8)
            if material=='floor':
                image[:]=color
            else:
                for u in (70,200):
                    cv2.line(image,(u,6),(u,233),color,5)
            r=engine.LaneDetector(engine.LaneConfig(max_frame_age_s=None)).process(bev_fixture(image))
            semantics.append(dict(color=name,material=material,valid=r.valid,confidence=r.confidence,
                false_valid=bool(r.valid and material!='paint_control'),reasons=r.reason_codes,
                hsv=cv2.cvtColor(np.uint8([[color]]),cv2.COLOR_BGR2HSV)[0,0].tolist()))
    arcs=[]
    for radius in (120,140,160,180,220,300,700):
        for width in (60,100,130):
            image,circle=arc_scene(radius,width)
            r=engine.LaneDetector(engine.LaneConfig(max_frame_age_s=None)).process(bev_fixture(image))
            error=None
            if r.valid:
                p=r.selected_pair.centreline
                pts=p.samples(100)
                error=float(np.max(np.abs(np.linalg.norm(pts-np.asarray(circle[:2]),axis=1)-radius)))
            arcs.append(dict(radius_px=radius,width_px=width,valid=r.valid,limited=r.limited_support,
                centre_radial_error_px=error,confidence=r.confidence,reasons=r.reason_codes,
                fit_diagnostics=[asdict(b.diagnostics) for b in r.boundaries if b.diagnostics]))
    return dict(semantics=semantics,arcs=arcs,
        limitation='Paint/wall/glare controls have identical pixels; no image-only semantic disambiguation is possible.',
        semantic_false_valid=sum(r['false_valid'] for r in semantics),
        semantic_negative_scenes=sum(r['material']!='paint_control' for r in semantics))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    cv2.setNumThreads(1)
    r=run(a.baseline_dir)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print({k:v for k,v in r.items() if k not in ('arcs','semantics')})
    for row in r['arcs']:
        print({k:v for k,v in row.items() if k!='fit_diagnostics'})


if __name__=='__main__':
    main()
