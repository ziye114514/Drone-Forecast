import argparse,csv,json,time,hashlib,platform,sys
from pathlib import Path
from collections import deque
import cv2
import numpy as np
from .config import Config
from .raw_io import open_raw,raw_chunks
from .perception import RotorPerception
from .motion import ForecastBank,METHODS,BASE_METHODS,extrapolate

COLORS={'cv':(235,190,50),'ca':(170,170,170),'adaptive_history':(200,80,190),'adaptive_rotor':(60,220,70),'adaptive':(40,170,255)}

def draw_rotor(image,r,label=None):
    mode=r.get('mode','ellipse')
    color={'ellipse':(60,220,70),'arc':(240,140,220),'line':(40,230,240)}[mode]
    if mode=='line':
        angle=np.deg2rad(r['angle']);delta=np.array([np.cos(angle),np.sin(angle)])*r['major']/2
        center=np.array([r['cx'],r['cy']])
        cv2.line(image,tuple((center-delta).astype(int)),tuple((center+delta).astype(int)),color,2)
    else:cv2.ellipse(image,((r['cx'],r['cy']),(r['major'],r['minor']),r['angle']),color,2)
    if label:cv2.putText(image,label,(int(r['cx']),int(r['cy'])),0,.45,color,1)

def draw_preview(total,obs,result,trail):
    grey=np.uint8(np.clip(np.log1p(total)/4,0,1)*255)
    image=cv2.cvtColor(grey,cv2.COLOR_GRAY2BGR)
    for i,r in enumerate(obs.get('rotors',[])):
        draw_rotor(image,r,f"R{i+1}:{r.get('mode','ellipse')}")
    if result and result['active']:
        p=np.array(result['states']['cv']['position']).astype(int)
        trail.append(tuple(p))
        if len(trail)>1:cv2.polylines(image,[np.array(trail,dtype=np.int32)],False,(255,230,80),2)
        for name,state in result['states'].items():
            if name not in ('cv','adaptive'):continue
            line=np.array([extrapolate(state,x) for x in np.linspace(0,.4,21)],np.int32)
            cv2.polylines(image,[line],False,COLORS[name],2)
            cv2.circle(image,tuple(line[-1]),4,COLORS[name],-1)
    else:trail.clear()
    cv2.rectangle(image,(0,0),(image.shape[1],100),(15,15,15),-1)
    freq=obs.get('blade_pass_hz')
    text=f"t={obs['t']:.2f}s   rotors={len(obs.get('rotors',[]))}   blade-pass={freq:.0f} Hz" if freq else f"t={obs['t']:.2f}s   rotors={len(obs.get('rotors',[]))}   frequency unavailable"
    cv2.putText(image,text,(15,25),0,.65,(240,240,240),1)
    cv2.putText(image,'RAW only | blue: CV | orange: adaptive | forecast +0.4 s',(15,52),0,.6,(240,240,240),1)
    status='UPDATE' if result and result.get('rotor_learning_allowed') else 'FROZEN / HISTORY FALLBACK'
    cv2.putText(image,f"Geometry quality={obs.get('geometry_quality',0):.2f} | rotor learning: {status}",(15,80),0,.55,(230,230,230),1)
    return image

def run(args):
    cfg=Config.load(args.config)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    cfg.save(output/'config.json')
    raw=Path(args.raw).resolve();reader=open_raw(raw)
    height,width=reader.get_size()
    perception=RotorPerception(width,height,cfg);bank=ForecastBank(cfg)
    video=None;trail=deque(maxlen=80);processing=[];detections=0;frames=0;events_total=0
    cumulative_updates={m:0 for m in BASE_METHODS[2:]};previous_updates={m:0 for m in BASE_METHODS[2:]};last_track=None
    valid_frequency_frames=0
    mode_counts={mode:0 for mode in ('ellipse','arc','line')};learning_frames=0
    source_hash=hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob('*.py')):source_hash.update(f.read_bytes())
    metadata=dict(input_raw=str(raw),input_bytes=raw.stat().st_size,width=width,height=height,
            start_s=args.start,duration_s=args.duration,units='image pixels and seconds',
            time_basis='Metavision RawReader do_time_shifting=True',
            inference_external_inputs=[str(raw)],target_mass=None,target_size=None,wind_speed=None,
            camera_intrinsics=None,source_sha256=source_hash.hexdigest(),version='0.2.0',python=sys.version,
            sensor_model='Prophesee SDK supported RAW; no ground-truth boxes or reconstructed frames',
            frequency_interpretation='blade-passage proxy; mechanical RPM requires validated blade count and harmonics',
            pose_interpretation='uncalibrated undirected projected normal; line mode provides axis only; not 3D attitude',
            opencv_threads=cv2.getNumThreads(),geometry_modes=['ellipse','arc','line'])
    (output/'run_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    observations=(output/'observations.jsonl').open('w',encoding='utf-8')
    traces=(output/'forecasts.jsonl').open('w',encoding='utf-8')
    csvfile=(output/'predictions.csv').open('w',newline='',encoding='utf-8')
    writer=csv.DictWriter(csvfile,fieldnames=['anchor_time_s','track_id','method','horizon_s','x_px','y_px','observed'])
    writer.writeheader();wall_start=time.perf_counter();first=None;last=None;last_print=wall_start
    try:
        iterator=iter(raw_chunks(reader,args.start,args.duration,cfg.step_us))
        while True:
            began=time.perf_counter()
            try:t,events=next(iterator)
            except StopIteration:break
            obs,total=perception.detect(events,t);result=bank.step(obs)
            for mode,n in obs.get('observation_modes',{}).items():mode_counts[mode]+=n
            learning_frames+=int(bool(result and result.get('rotor_learning_allowed')))
            if result:
                if result['track_id']!=last_track:previous_updates={m:0 for m in BASE_METHODS[2:]}
                for name,n in result['online_updates'].items():
                    cumulative_updates[name]+=max(0,n-previous_updates[name]);previous_updates[name]=n
                last_track=result['track_id']
            valid_frequency_frames+=int(obs.get('blade_pass_hz') is not None)
            elapsed=time.perf_counter()-began;processing.append(elapsed)
            observations.write(json.dumps(obs)+'\n')
            traces.write(json.dumps(result if result else dict(t=t,active=False))+'\n')
            if result and result['active']:
                for name,state in result['states'].items():
                    for horizon in cfg.horizons:
                        p=extrapolate(state,horizon)
                        writer.writerow(dict(anchor_time_s=t,track_id=result['track_id'],method=name,horizon_s=horizon,x_px=p[0],y_px=p[1],observed=result['observed']))
            if args.preview:
                if video is None:
                    video=cv2.VideoWriter(str(output/'prediction.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),1e6/cfg.step_us,(width,height))
                    if not video.isOpened():raise RuntimeError('Could not create preview video')
                image=draw_preview(total,obs,result,trail);video.write(image)
                if frames%50==0:cv2.imwrite(str(output/'latest_preview.png'),image)
            first=t if first is None else first;last=t;frames+=1;events_total+=len(events);detections+=int(obs['detected'])
            if time.perf_counter()-last_print>8:
                print(json.dumps(dict(sensor_time_s=round(t,2),frames=frames,detected=detections,wall_s=round(time.perf_counter()-wall_start,1))),flush=True)
                last_print=time.perf_counter()
    finally:
        observations.close();traces.close();csvfile.close()
        if video is not None:video.release()
    wall=time.perf_counter()-wall_start
    summary=dict(frames=frames,event_count=events_total,detection_frames=detections,detection_fraction=detections/max(frames,1),
            sensor_start_s=first,sensor_end_s=last,wall_s=wall,
            processed_sensor_seconds=(last-first+cfg.step_us/1e6) if first is not None else 0,
            processing_median_ms=float(np.median(processing)*1000) if processing else None,
            processing_p95_ms=float(np.percentile(processing,95)*1000) if processing else None,
            timing_scope='decode + perception + filtering; disk output/video excluded from per-step timing; included in wall time',
            online_updates=cumulative_updates,valid_frequency_frames=valid_frequency_frames,
            rotor_geometry_counts=mode_counts,rotor_learning_allowed_frames=learning_frames)
    summary['realtime_factor']=summary['processed_sensor_seconds']/max(wall,1e-9)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

def main():
    p=argparse.ArgumentParser(description='RAW-only event drone image-plane forecasting')
    p.add_argument('--raw',required=True,help='Only sensor input: Prophesee .raw file')
    p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.)
    p.add_argument('--duration',type=float,default=0.,help='0 reads to end')
    p.add_argument('--config',help='Optional generic algorithm settings, no target calibration')
    p.add_argument('--preview',action='store_true')
    args=p.parse_args()
    if args.start<0 or args.duration<0:p.error('start and duration must be nonnegative')
    run(args)
