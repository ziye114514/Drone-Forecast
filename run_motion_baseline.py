"""RAW-only OpenCV motion reference. Activity is not a UAV classifier."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage2'),str(ROOT/'research/stage7')]
from activity_fallback_fixed import ActivityTracker
from models import MotionBank,predict,PARAMS
from event_drone_forecast.raw_io import open_raw,raw_chunks
from event_drone_forecast.rotor_sensing import PaperPerception
from event_drone_forecast.paper_config import PaperConfig

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--duration',type=float,default=20.);p.add_argument('--front',choices=['activity','rotor','both'],default='activity');a=p.parse_args()
    dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'observations_forecasts.jsonl').exists(),'Refuse to overwrite a run'
    cv2.setNumThreads(1);reader=open_raw(a.raw);height,width=reader.get_size()
    fronts={}
    if a.front in ['activity','both']:fronts['activity']=ActivityTracker(width,height)
    if a.front in ['rotor','both']:fronts['rotor']=PaperPerception(width,height,PaperConfig())
    banks={k:MotionBank() for k in fronts};started=time.perf_counter();times=[];count=0;event_count=0
    with (dest/'observations_forecasts.jsonl').open('w',encoding='utf-8') as f:
        for t,e in raw_chunks(reader,0.,a.duration,10000):
            tick=time.perf_counter();row=dict(t=t,frontends={});event_count+=len(e)
            for name,front in fronts.items():
                o=front.step(e,t) if name=='activity' else front.detect(e,t)[0]
                state=banks[name].step(t,o.get('center') if o['detected'] else None)
                forecasts=None if state is None else {m:{str(h):predict(s,h).tolist() for h in [.1,.2,.4]} for m,s in state.items()}
                row['frontends'][name]=dict(observation=o,states=state,forecasts=forecasts)
            f.write(json.dumps(row,allow_nan=False)+'\n');times.append((time.perf_counter()-tick)*1000);count+=1
    wall=time.perf_counter()-started
    sources=[Path(__file__),ROOT/'research/stage7/models.py',ROOT/'research/stage2/activity_fallback.py',ROOT/'research/stage2/activity_fallback_fixed.py',*(ROOT/'src/event_drone_forecast').glob('*.py')]
    meta=dict(raw=str(Path(a.raw).resolve()),input_files=[str(Path(a.raw).resolve())],labels_in_inference=False,
        frontends=list(fronts),frames=count,events=event_count,duration_s=a.duration,wall_s=wall,
        processing_median_ms=float(np.median(times)),processing_p95_ms=float(np.percentile(times,95)),
        parameters=PARAMS,source_hashes={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources},
        scope='2D kinematic reference, not a pose/thrust estimator. Activity assumes a dominant compact moving object and does not verify UAV identity.',
        timing='Single offline RAW run; wall includes decode and JSON output. Streaming latency and long-run real-time guarantees are unverified.')
    (dest/'run_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
