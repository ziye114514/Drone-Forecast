"""Known flickering stripes and old79 nuisance controls; no RPM claim."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage141')]
from native_axis141 import NativeAxis,native79,PARAMS
OUT=ROOT.parent/'output/stage141_periodic_axis_20261024'
def main():
    OUT.mkdir(parents=True,exist_ok=True);results=[]
    for kind in ['tone','moving_tone','common_flicker','poisson','colored','burst','switch']:
        for seed in range(4):
            e,bbox,hz=native79.synthetic(kind,seed);moving=kind=='moving_tone'
            if kind in ['tone','moving_tone']:e['y']=48+(e['y']%2)*2
            observer=NativeAxis(128,96);valid=axis=0;last=None
            for tick in range(1,21):
                t=tick*.02;chunk=e[(e['t']>=round((t-.02)*1e6))&(e['t']<round(t*1e6))];q=observer.feed(chunk,t)
                # Moving support uses a current box translated with its known
                # synthetic velocity; no real-data truth enters this control.
                b=bbox.copy()
                if moving:b[[0,2]]+=(t-.4)*100
                last,_=observer.measure(q,dict(detected=True,confidence=.9,bbox=b),t,[100.,0.] if moving else [0.,0.]);valid+=last['qualified'];axis+=last['axis_valid']
            results.append(dict(kind=kind,seed=seed,known_flicker_hz=hz,qualified_windows=valid,line_windows=axis,final=last))
    assert any(r['line_windows'] for r in results if r['kind']=='tone'),'Known stationary periodic stripe never produced a line.'
    result=dict(parameters=PARAMS,results=results,scope='Known pixel flicker process and line, not real propeller/attitude/RPM truth. Nuisance false positives retained; no threshold tuning.')
    (OUT/'controls141.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8');print(json.dumps([dict(kind=k,qualified=sum(r['qualified_windows'] for r in results if r['kind']==k),lines=sum(r['line_windows'] for r in results if r['kind']==k)) for k in ['tone','moving_tone','common_flicker','poisson','colored','burst','switch']]),flush=True)
if __name__=='__main__':main()
