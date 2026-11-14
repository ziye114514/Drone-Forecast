"""Interleaved real saved input comparison; operator time, not detector FPS."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from rotor_support_front import estimate_region_translation


def main(output):
    import cv2
    cv2.setNumThreads(2)
    base=ROOT.parent/'output/rotor_pipeline_20261101';records=[]
    for clip in ['clear','side']:
        for a,b in [(0,8),(8,14),(14,23)]:
            with np.load(base/f'{clip}_bank/images/{a:04d}_evidence.npz') as z:
                previous=(z['on'].copy(),z['off'].copy());mask=z['native_arcs'].copy()
            with np.load(base/f'{clip}_bank/images/{b:04d}_evidence.npz') as z:
                current=(z['on'].copy(),z['off'].copy());support=z['native_arcs'].copy()
            for option in [False,True]:
                estimate_region_translation(previous,mask,current,current_support=support,integral_energy=option)
            for repeat in range(3):
                results={}
                for option in ([False,True] if repeat%2==0 else [True,False]):
                    start=time.perf_counter()
                    results[option]=estimate_region_translation(previous,mask,current,current_support=support,integral_energy=option)
                    records.append(dict(clip=clip,pair=[a,b],repeat=repeat,integral=option,ms=(time.perf_counter()-start)*1000))
                a0,b0=results[False],results[True]
                assert (a0 is None)==(b0 is None)
                if a0:assert (a0['dx'],a0['dy'])==(b0['dx'],b0['dy'])
    r=dict(records=records,all_six_saved_pairs_same_displacement=True,
        means_ms={str(v):float(np.mean([r['ms'] for r in records if r['integral']==v])) for v in [False,True]},
        core_pipeline_FPS_measured=False)
    Path(output).write_text(json.dumps(r,indent=2),'utf-8');print(json.dumps(r))


if __name__=='__main__':main(sys.argv[1])
