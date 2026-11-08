"""Run preserved reviewer functions without their machine-specific rv imports.

Provide --reviewer-dir with the evidence archive's p9_r09a.py/p10_r08.py.
Only original function definitions are compiled; image generation and geometry
equations are unchanged. Runtime imports/adapters use the selected local engine.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .availability_evaluation import engine_at
from .synthetic import bev_fixture, scene


def run(reviewer_dir, directory=None):
    engine=engine_at(directory)
    def detector(pkg, **kwargs):
        return engine.LaneDetector(engine.LaneConfig(**kwargs))
    namespace=dict(np=np,cv2=cv2,scene=scene,
        blank=lambda:np.full((240,320,3),35,np.uint8),
        frame=lambda pkg,image:bev_fixture(image),detector=detector,W=130.,CX=135.)
    paths=[reviewer_dir/'p9_r09a.py',reviewer_dir/'p10_r08.py']
    for path in paths:
        nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    # p10's scene overrides p9's imported scene; p9 clean needs the polynomial
    # renderer, so use a separate namespace for its original functions.
    original_scene=namespace['scene']
    p9=dict(np=np,cv2=cv2,scene=scene)
    nodes=[n for n in ast.parse(paths[0].read_text()).body if isinstance(n,ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(paths[0]),'exec'),p9)
    clutter=[]
    for kind,fn,counts in (('fragments','frags',(40,60,80,100)),('specks','specks',(100,200,300))):
        for count in counts:
            for seed in range(8):
                image=p9[fn](p9['clean'](),count,seed)
                r=detector(None,max_frame_age_s=None).process(bev_fixture(image))
                passed=p9['check'](r)
                clutter.append(dict(kind=kind,count=count,seed=seed,valid=r.valid,
                    original_midpoint_check=passed,reasons=r.reason_codes,support=r.usable_support_v_px,
                    image_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
    transitions=[]
    for dashed in (False,True):
        for radius in (120,220,400,800,2000):
            for vt in (40,70,100,130,160,190,215):
                for direction in (1,-1):
                    image=original_scene(radius,vt,direction,dashed)
                    r=namespace['evaluate'](None,image,radius,vt,direction)
                    transitions.append(dict(radius=radius,transition=vt,direction=direction,dashed=dashed,
                        image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),**r))
    return dict(baseline_directory=str(directory) if directory else None,
        original_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        adaptation='Original functions only; portable imports, BEV fixture and selected detector adapters.',
        clutter=clutter,transitions=transitions,
        solid_valid=sum(r['validity']=='valid' for r in transitions if not r['dashed']),
        dashed_valid=sum(r['validity']=='valid' for r in transitions if r['dashed']),
        incorrect_valid=sum(r['validity']=='valid' and max(r['centre_err'],r['bound_err'])>2.5 for r in transitions),
        outside_domain_errors=sum(r['validity']=='valid' and r['limited'] and not r['nan_out'] for r in transitions))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviewer-dir',type=Path,required=True)
    p.add_argument('--baseline-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    cv2.setNumThreads(1)
    r=run(a.reviewer_dir,a.baseline_dir)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print({k:v for k,v in r.items() if k not in ('clutter','transitions')})


if __name__=='__main__':
    main()
