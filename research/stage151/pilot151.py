"""RAW observation probe with frozen existing boxes; not end-to-end inference."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from event_drone_forecast.raw_io import open_raw,raw_chunks
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from sdkflow151 import FlowObservation,MotionState,PARAMS

OUT=ROOT.parent/'output/stage151_event_motion_20261024'
OUT.mkdir(exist_ok=True)
cases=[('side','E:/dataset/train/6/media/event.raw',110.,9.5),
       ('train2','E:/dataset/train/2/aligned/event.raw',40.,6.)]
for name,raw,start,duration in cases:
    saved=[json.loads(x) for x in (ROOT.parent/'output/stage149_inertia_scale_20261024'/name/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    rows={round(r['t'],6):r for r in saved}
    reader=open_raw(raw);h,w=reader.get_size();flow=FlowObservation(w,h);state=MotionState()
    noise=ActivityNoiseFilterAlgorithm(w,h,10000);buffer=noise.get_empty_output_buffer()
    results=[];cost=[]
    for t,e in raw_chunks(reader,start-.1,duration+.1,20000):
        noise.process_events(e,buffer);clean=buffer.numpy().copy()
        r=rows.get(round(t,6))
        if r is None:continue
        tick=time.perf_counter();o=flow.step(clean,t,r['body'])
        result=state.step(t,r['body'],o,r['coefficients'].get('optimized_geometry'),
            r['predictions'].get('optimized_geometry',{}),r['predictor']['structure_supported'])
        cost.append((time.perf_counter()-tick)*1000)
        results.append(dict(t=t,event_motion=result,body=r['body'],valid=r['valid'],
            coefficients=r['coefficients'],predictions=dict(r['predictions'],event_motion=result['predictions'])))
        if len(results)%100==0:print(json.dumps(dict(case=name,frames=len(results),valid=sum(x['event_motion']['active'] for x in results),ms=float(np.median(cost)))),flush=True)
    (OUT/(name+'_observation_probe.jsonl')).write_text('\n'.join(json.dumps(x,allow_nan=False) for x in results)+'\n',encoding='utf-8')
    print(json.dumps(dict(case=name,frames=len(results),active=sum(x['event_motion']['active'] for x in results),ms=float(np.median(cost)),not_end_to_end=True)),flush=True)
(OUT/'frozen_parameters.json').write_text(json.dumps(PARAMS,indent=2),encoding='utf-8')
