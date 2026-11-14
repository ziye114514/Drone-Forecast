"""RAW-only native10 geometry, rotor sensing, projected physics and causal model weighting."""
from pathlib import Path
from collections import deque
import argparse,hashlib,importlib.util,json,sys,time
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage28'),str(ROOT/'research/stage29'),str(ROOT/'research/stage9'),str(ROOT/'src')]
import axis28,select29
from recovery import RecoveryFrontend
from event_drone_forecast.raw_io import open_raw,raw_chunks
from event_drone_forecast.rotor_sensing import RotorSensing
from event_drone_forecast.paper_config import PaperConfig

spec=importlib.util.spec_from_file_location('visual_only_runner24',ROOT/'core/run_research_forecast.py')
visual=importlib.util.module_from_spec(spec);spec.loader.exec_module(visual)
json_text=visual.json_text


def fingerprints():
    result=visual.fingerprints()
    for p in [Path(__file__),ROOT/'research/stage28/axis28.py',ROOT/'research/stage27/physics27.py',ROOT/'research/stage29/select29.py']:
        result[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    result['run_research_forecast.py']=hashlib.sha256((ROOT/'core/run_research_forecast.py').read_bytes()).hexdigest()
    return result


def run(raw,output,start=0.,duration=0.,primary='adaptive_all_mix',preview=False):
    raw=Path(raw).resolve();output=Path(output).resolve();protected=Path('D:/yolov10-main')
    if raw.is_relative_to(protected) or output.is_relative_to(protected):raise ValueError('Protected directory excluded')
    if min(start,duration)<0 or round(start*1e6)%10000 or round(duration*1e6)%10000:raise ValueError('Use nonnegative10ms start/duration')
    if primary not in select29.MODELS:raise ValueError(primary)
    output.mkdir(parents=True,exist_ok=False);hashes=fingerprints();cfg=PaperConfig();stat=raw.stat()
    protocol=dict(raw=str(raw),raw_size_bytes=stat.st_size,raw_mtime_ns=stat.st_mtime_ns,start=start,duration=duration,source_hashes=hashes,
        primary=primary,step_us=10000,axis='canonical_static',parameters=select29.PARAMS,physics_parameters=axis28.prior.PARAMS,
        inputs='RAW only. No labels, cache, manifests, future frames or known mass/size/wind/RPM read.',
        cold_start=True,preview=preview,scope='Image-plane research. Actual ellipse/frequency features enter physics experts, online weights receive delayed current detector position only. Coefficients effective pixels, not physical identification. Image-up/non-inverted prior; no general camera-roll calibration.')
    (output/'protocol.json').write_text(json_text(protocol),encoding='utf-8')
    began=time.perf_counter();reader=open_raw(raw);h,w=reader.get_size();front=RecoveryFrontend(w,h,'neighbor10')
    sensor=RotorSensing(cfg);history=deque();physics=axis28.Predictor('canonical_static');selector=select29.Selector();video=None
    if preview:
        video=cv2.VideoWriter(str(output/'forecasts.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),100.,(w,h))
        if not video.isOpened():raise RuntimeError('Cannot open preview')
    timing=[];n=detected=valid=active=0;last_print=began
    try:
        with (output/'observations_forecasts.jsonl').open('x',encoding='utf-8') as stream:
            for t,events in raw_chunks(reader,start,duration,10000):
                tick=time.perf_counter();obs,_=front.detect(events,t)
                stride=max(1,int(np.ceil(len(events)/max(1,cfg.max_buffer_events//6))))
                history.append((t,events[::stride].copy()))
                while history and history[0][0]<t-cfg.frequency_window_us/1e6-.03:history.popleft()
                sensed=sensor.step(obs,history,t);row=dict(t=t,event_count=int(len(events)),observation=obs,sensing=sensed)
                q,_=physics.step(row);q=selector.step(q,obs);elapsed=(time.perf_counter()-tick)*1000
                row.update(forecasting=q,primary=primary,primary_prediction=q['predictions'].get(primary),processing_ms=elapsed)
                stream.write(json_text(row)+'\n');n+=1;detected+=obs['detected'];valid+=q['valid'];active+=q['feature_diagnostic']['active'];timing.append(elapsed)
                if video is not None:
                    frame=visual.draw(events,w,h,obs,q,t,'native10+axis+frequency',primary)
                    if q['valid']:
                        for horizon,point in q['predictions']['adaptive_axis_freq'].items():
                            cv2.circle(frame,tuple(np.rint(point).astype(int)),7,(160,65,170),1)
                    weights=q['selection_diagnostic']['weights']['all']
                    cv2.putText(frame,f'physics={q["feature_diagnostic"]["active"]} | weights CV={weights["cv"]:.2f} thrust={weights["adaptive_axis_freq"]:.2f}',(16,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(50,65,85),2)
                    video.write(frame)
                if time.perf_counter()-last_print>15:
                    print(json_text(dict(t=t,frames=n,detected=detected,forecasts=valid,physics_active=active)),flush=True);last_print=time.perf_counter()
    finally:
        if video is not None:video.release()
    summary=dict(frames=n,detected=int(detected),valid_forecasts=int(valid),physics_active_ticks=int(active),wall_including_decode_preview_s=time.perf_counter()-began,
        processing_median_ms=float(np.median(timing)) if timing else None,processing_p95_ms=float(np.percentile(timing,95)) if timing else None,
        sources_unchanged=fingerprints()==hashes,labels_in_inference=False,width=w,height=h,video_source_fps=100,
        scope='Video100fps follows RAW timestamps, not throughput. All15 predictors saved. Physics weight may duplicate CV when features inactive; no independent thrust/RPM/pose truth.')
    (output/'completion.json').write_text(json_text(summary),encoding='utf-8');print(json_text(summary),flush=True);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--primary',choices=select29.MODELS,default='adaptive_all_mix');p.add_argument('--preview',action='store_true');a=p.parse_args()
    run(a.raw,a.output,a.start,a.duration,a.primary,a.preview)
