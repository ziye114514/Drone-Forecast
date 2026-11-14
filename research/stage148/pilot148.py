"""Observation-only real RAW pilot with frozen current147 body detections.
No new NN run, no future box input. Does not count as end-to-end reproduction.
"""
from pathlib import Path
from collections import Counter
import json,sys,time,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent
OUT=WORK/'output/stage148_temporal_rotors_20261024'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage148'),str(ROOT/'research/stage129')]
from temporal148 import BranchObserver
from polarity import counts
from event_drone_forecast.raw_io import open_raw,raw_chunks
def js(value):return json.dumps(value,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())
def main():
    p=argparse.ArgumentParser();p.add_argument('--suffix',default='');p.add_argument('--case',choices=['clear','side']);a=p.parse_args()
    OUT.mkdir(exist_ok=True);summary={}
    for name,start,duration in [('clear',26.,1.5),('side',116.,2.5)]:
        if a.case and a.case!=name:continue
        old=WORK/f'output/stage147_curvature_split_20261024/{name}_paired/predictions.jsonl'
        rows={round(r['t'],6):r for r in map(json.loads,old.read_text().splitlines())}
        reader=open_raw('E:/dataset/train/6/media/event.raw');h,w=reader.get_size();observer=BranchObserver(w,h);result=[];times=[]
        for t,e in raw_chunks(reader,start,duration,20000):
            row=rows[round(t,6)];prepared=counts(e,w,h);prepared['on']/=2;prepared['off']/=2
            tic=time.perf_counter();m=observer.step(e,t,prepared,row['body'],row['coefficients'].get('optimized_geometry'),row['predictions'].get('optimized_geometry',{}));times.append((time.perf_counter()-tic)*1000)
            if name=='clear' and abs(t-26.84)<1e-8:
                (OUT/'fragment_debug148.json').write_text(js(dict(prior=observer.front.guide.quad,audit=observer.front.fragment_audit)),encoding='utf-8')
            result.append(dict(t=t,rotor_branch=m,source='cached current147 body only; RAW observation replay'))
        (OUT/f'pilot_{name}{a.suffix}.jsonl').write_text('\n'.join(js(r) for r in result)+'\n',encoding='utf-8')
        summary[name]=dict(ticks=len(result),states=dict(Counter(r['rotor_branch']['geometry']['state'] for r in result)),
            physical_ticks=sum(r['rotor_branch']['thrust_proxy']['valid'] for r in result),
            matched=sum(r['rotor_branch']['temporal148']['matched'] for r in result),
            frequency_reasons=dict(Counter(q.get('reason','none') for r in result for q in r['rotor_branch']['rates'])),
            median_ms=float(np.median(times)),end_to_end=False)
        print(js(dict(case=name,**summary[name])),flush=True)
    (OUT/f'pilot_summary148{a.suffix}.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
if __name__=='__main__':main()
