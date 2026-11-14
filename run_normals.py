"""Experimental normal ablation; RAW-only, no invented camera calibration."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'research/stage3'))
from raw_pipeline import AuditedPerception,ConsensusBank
from event_drone_forecast.rotor_sensing import RotorSensing
from event_drone_forecast.paper_bank import PaperForecastBank
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.raw_io import open_raw,raw_chunks

def source_hashes():
    files=[Path(__file__),ROOT/'research/stage3/measurement.py',ROOT/'research/stage3/raw_pipeline.py',*(ROOT/'src/event_drone_forecast').glob('*.py')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--start',type=float,default=0);parser.add_argument('--duration',type=float,default=30);parser.add_argument('--config');a=parser.parse_args()
    cfg=PaperConfig.load(a.config);folder=Path(a.output)
    if (folder/'run_manifest.json').exists():raise RuntimeError('Existing run; use new directory')
    folder.mkdir(parents=True,exist_ok=True);raw=Path(a.raw).resolve();reader=open_raw(raw);height,width=reader.get_size()
    frontend=AuditedPerception(width,height,cfg);sensor=RotorSensing(cfg)
    banks={'frozen':PaperForecastBank(cfg),'consensus':ConsensusBank(cfg,False),'consensus_audit':ConsensusBank(cfg,True)}
    meta=dict(version='stage3-normal-ablation',raw=str(raw),inference_inputs=[str(raw)],start_s=a.start,duration_s=a.duration,
        source_hashes=source_hashes(),camera_matrix_supplied=cfg.camera_matrix is not None,
        normal_scope='Calibrated circle candidates only with suppliedK; otherwise provisional weak-perspective image-axis consistency.',
        positions_and_rpm_shared=True,independent_pose_truth=False,selected_default_upgrade=False)
    (folder/'run_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');cfg.save(folder/'config.json')
    handles={};count={name:dict(active=0,physics=0,accepted_normals=0) for name in banks};samples=[];frames=event_count=0
    for name in banks:
        dest=folder/name;dest.mkdir(exist_ok=True);handles[name]=(dest/'forecasts.jsonl').open('w',encoding='utf-8')
    started=time.perf_counter();lastprint=started
    try:
        with (folder/'observations.jsonl').open('w',encoding='utf-8') as f:
            for t,e in raw_chunks(reader,a.start,a.duration,cfg.step_us):
                before=time.perf_counter();o,total=frontend.detect(e,t);s=sensor.step(o,frontend.history,t)
                f.write(json.dumps(dict(t=t,observation=o,sensing=s),allow_nan=False)+'\n')
                reference=None
                for name,bank in banks.items():
                    result=bank.step(o,s)
                    if result is None:result=dict(t=t,active=False,states={},physics_active=False)
                    handles[name].write(json.dumps(result,allow_nan=False)+'\n')
                    c=count[name];c['active']+=int(result['active']);c['physics']+=int(result['physics_active'])
                    c['accepted_normals']+=int(all(d['accepted'] for d in result.get('normal_measurements',{}).values()) and bool(result.get('normal_measurements')))
                    if reference is None:reference=result
                    else:
                        assert result['active']==reference['active']
                        if result['states']:
                            for key in ('position','velocity','acceleration'):np.testing.assert_array_equal(result['states']['cv'][key],reference['states']['cv'][key])
                frames+=1;event_count+=len(e);samples.append((time.perf_counter()-before)*1000)
                if time.perf_counter()-lastprint>12:
                    print(json.dumps(dict(t=t,frames=frames,arms=count)),flush=True);lastprint=time.perf_counter()
    finally:
        for f in handles.values():f.close()
    summary=dict(frames=frames,events=event_count,arms=count,wall_s=time.perf_counter()-started,
        processing_median_ms=float(np.median(samples)),processing_p95_ms=float(np.percentile(samples,95)),
        updates={n:b.models['paper_adaptive'].updates for n,b in banks.items()},camera_matrix_supplied=cfg.camera_matrix is not None,
        cv_states_identical_verified=True)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
