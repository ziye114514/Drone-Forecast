"""One RAW input -> current rotor observation + causal2D forecasts; research entry."""
from pathlib import Path
from collections import deque
import argparse,hashlib,importlib.util,json,time
import cv2,numpy as np

ROOT=Path(__file__).resolve().parent
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

windows=module('research_windows21',ROOT/'research/stage21/windows21.py')
robust=module('research_robust23',ROOT/'research/stage23/robust23.py')
from event_drone_forecast.raw_io import open_raw,raw_chunks
from event_drone_forecast.rotor_sensing import RotorSensing
from event_drone_forecast.paper_config import PaperConfig


def json_text(x):
    def scalar(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        raise TypeError(type(v).__name__)
    return json.dumps(x,ensure_ascii=False,allow_nan=False,default=scalar)


def fingerprints():
    files=[Path(__file__),ROOT/'research/stage21/windows21.py',ROOT/'research/stage23/robust23.py',ROOT/'research/stage22/predict22.py',
        ROOT/'research/stage14/motion.py',ROOT/'research/stage18/frontend18.py',ROOT/'research/stage17/frontend.py',
        ROOT/'research/stage9/recovery.py',ROOT/'research/stage8/polarity.py',*(ROOT/'src/event_drone_forecast').glob('*.py')]
    return {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def draw(events,w,h,observation,forecast,t,frontend,primary):
    pixels=events['y'].astype(np.int32)*w+events['x'];count=np.bincount(pixels,minlength=w*h).reshape(h,w)
    grey=np.uint8(255-230*np.clip(np.log1p(count)/np.log(8),0,1));frame=cv2.cvtColor(grey,cv2.COLOR_GRAY2BGR)
    if observation['detected']:
        b=np.rint(observation['bbox']).astype(int);cv2.rectangle(frame,tuple(b[:2]),tuple(b[2:]),(40,150,30),2)
        cv2.drawMarker(frame,tuple(np.rint(observation['center']).astype(int)),(40,150,30),cv2.MARKER_CROSS,14,2)
    if forecast['valid']:
        for model,color in [('cv',(200,105,30)),(primary,(25,125,225))]:
            c=forecast['coefficients'][model];points=[c['position']]+[forecast['predictions'][model][str(h)] for h in [.1,.2,.4]]
            for a,b in zip(points,points[1:]):cv2.line(frame,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),color,2)
            for horizon,p in forecast['predictions'][model].items():
                q=tuple(np.rint(p).astype(int));cv2.circle(frame,q,4,color,-1)
                cv2.putText(frame,horizon+'s',q,cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
    status='current' if forecast['anchor_has_current_measurement'] else 'no current measurement'
    cv2.putText(frame,f'{t:.2f}s | {frontend} | detection={observation["detected"]} | forecast={forecast["valid"]} | {status}',(16,h-24),cv2.FONT_HERSHEY_SIMPLEX,.55,(50,65,85),2)
    return frame


def run(raw,output,start=0.,duration=0.,frontend='native20',primary='trusted_acc',rotor_features=False,preview=False):
    raw=Path(raw).resolve();output=Path(output).resolve();protected=Path('D:/yolov10-main')
    if raw.is_relative_to(protected) or output.is_relative_to(protected):raise ValueError('Protected directory excluded')
    if min(start,duration)<0 or round(start*1e6)%20000 or round(duration*1e6)%20000:raise ValueError('Start/duration must be nonnegative20ms multiples')
    if frontend not in windows.METHODS or primary not in robust.MODELS:raise ValueError('Unknown research mode')
    output.mkdir(parents=True,exist_ok=False);hashes=fingerprints();stat=raw.stat()
    cfg=PaperConfig(step_us=20000);protocol=dict(raw=str(raw),raw_size_bytes=stat.st_size,raw_mtime_ns=stat.st_mtime_ns,start=start,duration=duration,
        frontend=frontend,primary=primary,source_hashes=hashes,rotor_features=rotor_features,preview=preview,step_us=20000,
        predictor_parameters=robust.PARAMS,frequency_parameters=dict(window_us=cfg.frequency_window_us,update_steps=cfg.frequency_update_steps,blade_count_assumed=cfg.blade_count),
        inputs='RAW only; no labels, manifests, cached detections or future frames read by inference.',warmup='Up to100ms before start; geometry initialized exactly like stage21; predictions start strictly after start.',
        scope='Experimental image-plane forecasts from position history. Optional rotor frequency/ellipse features are diagnostic, not used by these predictors. No mechanical RPM/thrust/pose truth.')
    (output/'protocol.json').write_text(json_text(protocol),encoding='utf-8')
    began=time.perf_counter();reader=open_raw(raw);h,w=reader.get_size();front=windows.Observer(w,h,frontend);predictor=robust.Predictor()
    sensing=RotorSensing(cfg) if rotor_features else None;history=deque();video=None
    if preview:
        video=cv2.VideoWriter(str(output/'forecasts.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50.,(w,h))
        if not video.isOpened():raise RuntimeError('Cannot open video writer')
    begin=max(0.,start-.1);length=duration+(start-begin) if duration else 0.;n=detected=published=four_frequencies=0;timing=[];last_print=began
    try:
        with (output/'observations_forecasts.jsonl').open('x',encoding='utf-8') as stream:
            for t,events in raw_chunks(reader,begin,length,20000):
                if round(t*1e6)%20000:break
                tick=time.perf_counter();ready=t-begin>=.1-1e-8
                q=front.step(events,t) if ready else dict(observation=dict(t=t,detected=False,rotors=[]),audit={},candidates=[])
                observation=q['observation'];sensed=None
                if sensing is not None:
                    stride=max(1,int(np.ceil(len(events)/max(1,cfg.max_buffer_events//6))))
                    history.append((t,events[::stride].copy()))
                    while history and history[0][0]<t-cfg.frequency_window_us/1e6-.03:history.popleft()
                    sensed=sensing.step(observation,history,t)
                if t<=start+1e-8:continue
                forecast,_=predictor.step(t,observation['center'] if observation['detected'] else None)
                elapsed=(time.perf_counter()-tick)*1000;timing.append(elapsed)
                row=dict(t=t,observation=observation,forecasting=forecast,primary=primary,
                    primary_prediction=forecast['predictions'].get(primary),rotor_sensing=sensed,frontend_audit=q['audit'],processing_ms=elapsed)
                stream.write(json_text(row)+'\n');n+=1;detected+=observation['detected'];published+=forecast['valid']
                if sensed is not None:four_frequencies+=sensed['four_frequencies']
                if video is not None:video.write(draw(events,w,h,observation,forecast,t,frontend,primary))
                if time.perf_counter()-last_print>15:print(json_text(dict(t=t,frames=n,detected=detected,forecasts=published)),flush=True);last_print=time.perf_counter()
    finally:
        if video is not None:video.release()
    summary=dict(frames=n,detected=detected,forecast_anchors=published,ticks_four_frequencies_internal=four_frequencies,
        wall_including_open_seek_decode_write_preview_s=time.perf_counter()-began,
        processing_median_ms=float(np.median(timing)) if timing else None,processing_p95_ms=float(np.percentile(timing,95)) if timing else None,
        source_unchanged=fingerprints()==hashes,labels_in_inference=False,width=w,height=h,
        scope='Video50fps follows RAW timestamps, not processing throughput. Forecasts are image positions, not measured future trajectory. Optional frequencies are provisional.')
    (output/'completion.json').write_text(json_text(summary),encoding='utf-8');print(json_text(summary),flush=True);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--frontend',choices=list(windows.METHODS),default='native20');p.add_argument('--primary',choices=robust.MODELS,default='trusted_acc')
    p.add_argument('--rotor-features',action='store_true');p.add_argument('--preview',action='store_true');a=p.parse_args()
    run(a.raw,a.output,a.start,a.duration,a.frontend,a.primary,a.rotor_features,a.preview)
