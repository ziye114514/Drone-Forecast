import argparse,csv,hashlib,json,sys,time
from pathlib import Path
from collections import deque
import cv2,numpy as np
from .paper_config import PaperConfig
from .raw_io import open_raw,raw_chunks
from .rotor_sensing import PaperPerception,RotorSensing
from .paper_bank import PaperForecastBank
from .motion import extrapolate


def preview(total,obs,result):
    image=cv2.cvtColor(np.uint8(np.clip(np.log1p(total)/4,0,1)*255),cv2.COLOR_GRAY2BGR)
    for r in obs.get('rotors',[]):
        color=(40,220,70) if r.get('observed_now',False) else (140,140,140)
        cv2.ellipse(image,((r['cx'],r['cy']),(r['major'],r['minor']),r['angle']),color,2)
        hz=r.get('blade_pass_hz');label=f"ID{r.get('rotor_id','?')} {hz:.0f}Hz" if hz else f"ID{r.get('rotor_id','?')} warming"
        cv2.putText(image,label,(int(r['cx']),int(r['cy'])),0,.45,color,1)
    if result and result['active']:
        for name,color in [('cv',(235,190,50)),('paper_nominal',(210,90,220)),('paper_fused',(40,170,255))]:
            curve=np.array([extrapolate(result['states'][name],h) for h in np.linspace(0,.4,25)],np.int32)
            cv2.polylines(image,[curve],False,color,2);cv2.circle(image,tuple(curve[-1]),4,color,-1)
    cv2.rectangle(image,(0,0),(image.shape[1],82),(15,15,15),-1)
    status='THRUST ROLLOUT' if result and result.get('physics_active') else 'CV FALLBACK / SIGNAL WARMUP'
    cv2.putText(image,f"t={obs['t']:.2f}s | four visible={obs.get('four_visible')} | {status}",(12,25),0,.6,(245,245,245),1)
    cv2.putText(image,'Blue: CV | Pink: paper-model | Orange: gated fusion | future +0.4s',(12,52),0,.55,(230,230,230),1)
    return image


def run(args):
    cfg=PaperConfig.load(args.config);out=Path(args.output).resolve()
    if (out/'run_manifest.json').exists():raise RuntimeError('Existing run; choose a new output directory.')
    out.mkdir(parents=True,exist_ok=True);cfg.save(out/'config.json')
    reader=open_raw(args.raw);height,width=reader.get_size()
    frontend=PaperPerception(width,height,cfg);sensing=RotorSensing(cfg);bank=PaperForecastBank(cfg)
    source=hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):source.update(path.read_bytes())
    manifest=dict(version='0.3.0',input_raw=str(Path(args.raw).resolve()),input_bytes=Path(args.raw).stat().st_size,
        width=width,height=height,start_s=args.start,duration_s=args.duration,inference_external_inputs=[str(Path(args.raw).resolve())],
        source_sha256=source.hexdigest(),units='image pixels, seconds, provisional blade-passage Hz',
        camera_matrix_supplied=cfg.camera_matrix is not None,physical_scale_verified=False,
        normal_mode='calibrated-circle cone' if cfg.camera_matrix is not None else 'weak-perspective projected thrust axis',
        defaults_are_assumptions=dict(blade_count=cfg.blade_count,image_up=cfg.image_up,
                                     nominal_gravity_px_s2=cfg.nominal_gravity_px_s2),
        metric_output=False,future_input_policy='No future RPM/pose/labels; current rotor speed held during rollout.',
        motor_differential_mode='Stable image-left/right proxy; body motor correspondence and yaw are uncalibrated.',
        physical_parameter_fields='Optional metric parameters are for the separately tested metric kernel; this CLI outputs pixels only.',
        reference_paper='2604.18289v1; explicit equations implemented with documented image-space adaptation')
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    obsfile=(out/'observations.jsonl').open('w',encoding='utf-8');traces=(out/'forecasts.jsonl').open('w',encoding='utf-8')
    csvfile=(out/'predictions.csv').open('w',newline='',encoding='utf-8');writer=csv.writer(csvfile)
    writer.writerow(['anchor_time_s','method','horizon_s','x_px','y_px','physics_active','four_visible'])
    count=detected=event_count=four=physics=0;latencies=[];video=None;begin=time.perf_counter();last_print=begin;first=last=None
    try:
        iterator=iter(raw_chunks(reader,args.start,args.duration,cfg.step_us))
        while True:
            started=time.perf_counter()
            try:t,events=next(iterator)
            except StopIteration:break
            obs,total=frontend.detect(events,t);rotors=sensing.step(obs,frontend.history,t)
            obs['detected_rotors']=obs['rotors'];obs['rotors']=rotors['rotors']
            obs.update({k:v for k,v in rotors.items() if k!='rotors'})
            obs['frequency_quality']=rotors['frequency_quality'];obs['frequency_squared_relative']=rotors['thrust_ratio'] or 1.
            obs['rotor_frequencies_hz']=[r['blade_pass_hz'] if r['frequency_valid'] else None for r in rotors['rotors']]
            obs['frequency_diagnostics']=[dict(rotor_id=r['rotor_id'],frequency_valid=r['frequency_valid'],
                phase_coherence=r.get('phase_coherence'),harmonic_ambiguity=r.get('harmonic_ambiguity')) for r in rotors['rotors']]
            obs['blade_pass_hz']=float(np.mean([r['blade_pass_hz'] for r in rotors['rotors'] if r['frequency_valid']])) if any(r['frequency_valid'] for r in rotors['rotors']) else None
            result=bank.step(obs,rotors);latencies.append(time.perf_counter()-started)
            obsfile.write(json.dumps(obs,allow_nan=False)+'\n');traces.write(json.dumps(result or dict(t=t,active=False),allow_nan=False)+'\n')
            if result and result['active']:
                for name,state in result['states'].items():
                    for horizon in cfg.horizons:
                        p=extrapolate(state,horizon);writer.writerow([t,name,horizon,*p,result['physics_active'],rotors['four_visible']])
            if args.preview:
                if video is None:video=cv2.VideoWriter(str(out/'prediction.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),1e6/cfg.step_us,(width,height))
                video.write(preview(total,obs,result))
            count+=1;detected+=int(obs['detected']);event_count+=len(events);four+=int(rotors['four_visible']);physics+=int(bool(result and result['physics_active']))
            first=t if first is None else first;last=t
            if time.perf_counter()-last_print>8:
                print(json.dumps(dict(sensor_time_s=round(t,2),frames=count,detected=detected,four_visible=four,physics_active=physics)),flush=True);last_print=time.perf_counter()
    finally:
        obsfile.close();traces.close();csvfile.close()
        if video is not None:video.release()
    summary=dict(frames=count,event_count=event_count,detection_frames=detected,four_visible_frames=four,
        physics_active_frames=physics,processed_sensor_seconds=last-first+cfg.step_us/1e6 if first is not None else 0,
        wall_s=time.perf_counter()-begin,processing_median_ms=float(np.median(latencies)*1000),
        processing_p95_ms=float(np.percentile(latencies,95)*1000),
        effective_parameters=bank.models['paper_adaptive'].parameters.tolist(),physical_updates=bank.models['paper_adaptive'].updates,
        metric_parameter_recovery_claimed=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description='Paper-oriented RAW-only thrust-axis drone forecasting')
    parser.add_argument('--raw',required=True);parser.add_argument('--output',required=True);parser.add_argument('--config')
    parser.add_argument('--start',type=float,default=0.);parser.add_argument('--duration',type=float,default=0.)
    parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    if args.start<0 or args.duration<0:parser.error('start and duration must be nonnegative')
    run(args)
