"""Experimental RAW-only integration; frozen original runtime is unchanged."""
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage2'))
from activity_fallback_fixed import ActivityTracker
from event_drone_forecast.raw_io import open_raw,raw_chunks
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.rotor_sensing import PaperPerception,RotorSensing
from event_drone_forecast.paper_bank import PaperForecastBank
from event_drone_forecast.motion import extrapolate

def source_hashes():
    files=[Path(__file__),ROOT/'research/stage2/activity_fallback.py',ROOT/'research/stage2/activity_fallback_fixed.py',*(ROOT/'src/event_drone_forecast').glob('*.py')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}

def main():
    parser=argparse.ArgumentParser(description='RAW-only experimental hybrid position frontend with original rotor physics')
    parser.add_argument('--raw',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--start',type=float,default=0.);parser.add_argument('--duration',type=float,default=0.)
    parser.add_argument('--config');args=parser.parse_args();out=Path(args.output)
    if (out/'run_manifest.json').exists():raise RuntimeError('Existing run: choose a new output directory')
    out.mkdir(parents=True,exist_ok=True);cfg=PaperConfig.load(args.config);cfg.save(out/'config.json')
    raw=Path(args.raw).resolve();reader=open_raw(raw);height,width=reader.get_size()
    front=PaperPerception(width,height,cfg);activity=ActivityTracker(width,height);sensor=RotorSensing(cfg);bank=PaperForecastBank(cfg)
    manifest=dict(version='0.3.1-experimental',input_raw=str(raw),inference_external_inputs=[str(raw)],
        source_hashes=source_hashes(),start_s=args.start,duration_s=args.duration,width=width,height=height,
        rule='Four observed rotors and geometry_quality>.45: rotor position; otherwise activity-only position.',
        activity_estimates_pose_or_thrust=False,units='image pixels, seconds',calibration_supplied=cfg.camera_matrix is not None,
        physical_scale_verified=False,new_algorithm_claimed=False)
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    start=time.perf_counter();lastprint=start;durations=[];frames=events_count=detected=physics=0;source_counts={}
    with (out/'observations.jsonl').open('w',encoding='utf-8') as obsfile,(out/'forecasts.jsonl').open('w',encoding='utf-8') as predfile,(out/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as csvfile:
        writer=csv.writer(csvfile);writer.writerow(['anchor_t','method','horizon_s','x_px','y_px','position_source','physics_active'])
        for t,e in raw_chunks(reader,args.start,args.duration,cfg.step_us):
            begin=time.perf_counter();o,total=front.detect(e,t);a=activity.step(e,t);s=sensor.step(o,front.history,t)
            qualified=o['detected'] and o.get('observed_rotor_count',0)==4 and o.get('geometry_quality',0)>.45
            mixed=dict(o if qualified else a);mixed['position_source']='four_rotors' if qualified else 'activity_only'
            result=bank.step(mixed,s)
            if not qualified:
                # The activity observation carries no geometry; cached rotor
                # frequencies must not turn a position-only detection into thrust.
                assert not (result or {}).get('physics_active',False)
                assert all(not m.update_allowed for m in bank.models.values())
            record=dict(t=t,position_observation=mixed,original_rotor_observation=o,sensing=s)
            obsfile.write(json.dumps(record,allow_nan=False)+'\n')
            active=result is not None and result['active'];phys=bool((result or {}).get('physics_active',False))
            if result is None:result=dict(t=t,active=False,states={},physics_active=False)
            result['position_source']=mixed['position_source'];predfile.write(json.dumps(result,allow_nan=False)+'\n')
            if active:
                for name,state in result['states'].items():
                    for h in cfg.horizons:
                        p=extrapolate(state,h);writer.writerow([t,name,h,*p,mixed['position_source'],phys])
            frames+=1;events_count+=len(e);detected+=int(mixed['detected']);physics+=int(phys)
            source_counts[mixed['position_source']]=source_counts.get(mixed['position_source'],0)+int(mixed['detected'])
            durations.append((time.perf_counter()-begin)*1000)
            if time.perf_counter()-lastprint>12:
                print(json.dumps(dict(t=t,frames=frames,detected=detected,physics=physics)),flush=True);lastprint=time.perf_counter()
    summary=dict(version=manifest['version'],frames=frames,event_count=events_count,detected_frames=detected,physics_active_frames=physics,
        source_detection_frames=source_counts,wall_s=time.perf_counter()-start,processing_median_ms=float(np.median(durations)),
        processing_p95_ms=float(np.percentile(durations,95)),parameters={name:m.parameters.tolist() for name,m in bank.models.items()},
        physical_updates={name:m.updates for name,m in bank.models.items()},metric_parameter_recovery_claimed=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
