"""Experimental RAW-only whole-body axis forecast, no required labels or CAD.

Use the workstation's Python 3.11 / Metavision environment. The first useful
four-point view creates a model; before that body inference remains inactive.
See research/stage6/README.md for assumptions and validation limits.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage4'),str(ROOT/'research/stage6')]
from silhouette import BodyObserver,AxisForecast
from tracker import Skeleton
from event_drone_forecast.raw_io import open_raw,raw_chunks
from event_drone_forecast.rotor_sensing import PaperPerception
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.config import Config
from event_drone_forecast.motion import KinematicFilter

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=30.);a=p.parse_args()
    folder=Path(a.output)
    if (folder/'run_manifest.json').exists():raise RuntimeError('Use a new output directory')
    folder.mkdir(parents=True,exist_ok=True);reader=open_raw(a.raw);h,w=reader.get_size();front=PaperPerception(w,h,PaperConfig())
    observer=BodyObserver();four=Skeleton(body=False,partial=False);kf=KinematicFilter(Config(),2);predictor=AxisForecast()
    density=np.zeros((h,w),np.float32);seen=None;count=0;events=0;began=time.perf_counter();last=began
    files=[Path(__file__),ROOT/'research/stage6/silhouette.py',ROOT/'research/stage4/tracker.py',*(ROOT/'src/event_drone_forecast').glob('*.py')]
    meta=dict(raw=str(Path(a.raw).resolve()),start_s=a.start,duration_s=a.duration,inference_inputs=[str(Path(a.raw).resolve())],
        experimental=True,default_upgrade=False,requires_prior_four_point_view=True,head_tail_required=False,
        camera_matrix_supplied=False,calibrated_3d_pose=False,force_magnitude_measured=False,
        source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (folder/'run_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    with (folder/'observations_forecasts.jsonl').open('w',encoding='utf-8') as f:
        for i,(t,e) in enumerate(raw_chunks(reader,a.start,a.duration,10000)):
            o,total=front.detect(e,t);density*=np.exp(-.01/.02);density+=total;events+=len(e)
            if (i+1)%2:continue
            fm=four.step(t,o['rotors'],None,o.get('center'));body=observer.step(t,density,o,fm)
            z=body.get('center') if body['active'] and body.get('position_measured') else None
            if z is not None:seen=t
            if seen is not None and t-seen>.5:kf=KinematicFilter(Config(),2);seen=None
            state=kf.update(t,z,confidence=.75);active=state is not None and seen is not None and t-seen<=.3+1e-8
            forecasts={}
            if state is not None:
                pred=predictor.step(t,state,body,z)
                for method,acc in pred['accelerations'].items():
                    horizons={}
                    for horizon in [.1,.2,.4]:
                        horizons[str(horizon)]=(state[:2]+horizon*state[2:]+.5*horizon*horizon*np.array(acc)).tolist()
                    forecasts[method]=dict(active=bool(active),position=state[:2].tolist(),velocity=state[2:].tolist(),acceleration=acc,predictions_px=horizons)
            f.write(json.dumps(dict(t=t,observation=o,four_model=fm,body=body,forecasts=forecasts),allow_nan=False)+'\n');count+=1
            if time.perf_counter()-last>15:print(json.dumps(dict(t=t,frames=count,body_mode=body['mode'],active=bool(active))),flush=True);last=time.perf_counter()
    summary=dict(frames=count,events=events,wall_s=time.perf_counter()-began,model_memory=observer.memory,
                 interpretation='Image-plane axis and motion-derived acceleration proxy, not measured thrust or calibrated pose.')
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
