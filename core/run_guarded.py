"""RAW-only experimental rotor estimator with identity-bounded position fallback."""
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage2'))
from activity_fallback_fixed import ActivityTracker
from guarded_position import GuardedPosition
from event_drone_forecast.raw_io import open_raw,raw_chunks
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.rotor_sensing import PaperPerception,RotorSensing
from event_drone_forecast.paper_bank import PaperForecastBank
from event_drone_forecast.motion import extrapolate

def source_hashes():
    files=[Path(__file__),*(ROOT/'research/stage2'/name for name in ('activity_fallback.py','activity_fallback_fixed.py','guarded_position.py')),*(ROOT/'src/event_drone_forecast').glob('*.py')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--start',type=float,default=0.);parser.add_argument('--duration',type=float,default=0.);parser.add_argument('--config');args=parser.parse_args()
    out=Path(args.output)
    if (out/'run_manifest.json').exists():raise RuntimeError('Existing run: choose new output directory')
    out.mkdir(parents=True,exist_ok=True);cfg=PaperConfig.load(args.config);cfg.save(out/'config.json')
    raw=Path(args.raw).resolve();reader=open_raw(raw);height,width=reader.get_size()
    front=PaperPerception(width,height,cfg);activity=ActivityTracker(width,height);guard=GuardedPosition();sensor=RotorSensing(cfg);bank=PaperForecastBank(cfg)
    manifest=dict(version='0.3.2-guarded-experimental',input_raw=str(raw),inference_external_inputs=[str(raw)],
        source_hashes=source_hashes(),start_s=args.start,duration_s=args.duration,width=width,height=height,
        rule='Original measured rotor positions always retained. Activity only bridges a confirmed rotor target for <=0.3s and four-rotor confirmation age<=0.5s; past offset and spatial gate required.',
        activity_estimates_pose_or_thrust=False,activity_may_acquire_unknown_target=False,
        confirmed_identity_is_independent_ground_truth=False,units='image pixels, seconds',physical_scale_verified=False)
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    begin_run=time.perf_counter();lastprint=begin_run;durations=[];frames=event_count=detected=physics=0;sources={}
    with (out/'observations.jsonl').open('w',encoding='utf-8') as of,(out/'forecasts.jsonl').open('w',encoding='utf-8') as pf,(out/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as cf:
        writer=csv.writer(cf);writer.writerow(['anchor_t','method','horizon_s','x_px','y_px','position_source','physics_active'])
        for t,e in raw_chunks(reader,args.start,args.duration,cfg.step_us):
            begin=time.perf_counter();o,total=front.detect(e,t);a=activity.step(e,t);mixed=guard.step(o,a);s=sensor.step(o,front.history,t);result=bank.step(mixed,s)
            if mixed['position_source']!='rotor_observation':
                assert not (result or {}).get('physics_active',False)
                assert all(not model.update_allowed for model in bank.models.values())
            of.write(json.dumps(dict(t=t,position_observation=mixed,original_rotor_observation=o,sensing=s),allow_nan=False)+'\n')
            if result is None:result=dict(t=t,active=False,states={},physics_active=False)
            result['position_source']=mixed['position_source'];pf.write(json.dumps(result,allow_nan=False)+'\n')
            if result['active']:
                for method,state in result['states'].items():
                    for h in cfg.horizons:writer.writerow([t,method,h,*extrapolate(state,h),mixed['position_source'],result['physics_active']])
            frames+=1;event_count+=len(e);detected+=int(mixed['detected']);physics+=int(result['physics_active'])
            sources[mixed['position_source']]=sources.get(mixed['position_source'],0)+int(mixed['detected']);durations.append((time.perf_counter()-begin)*1000)
            if time.perf_counter()-lastprint>12:
                print(json.dumps(dict(t=t,frames=frames,detected=detected,physics=physics)),flush=True);lastprint=time.perf_counter()
    summary=dict(version=manifest['version'],frames=frames,event_count=event_count,detected_frames=detected,physics_active_frames=physics,
        source_detection_frames=sources,wall_s=time.perf_counter()-begin_run,processing_median_ms=float(np.median(durations)),processing_p95_ms=float(np.percentile(durations,95)),
        physical_updates={name:model.updates for name,model in bank.models.items()},metric_parameter_recovery_claimed=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
