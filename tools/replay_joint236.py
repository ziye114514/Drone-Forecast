"""CPU accuracy-only joint-phase bank, same frozen position/motion inputs."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,copy,argparse,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from joint_phase_bank import JointPhaseBank
from adaptive_response_observer import AdaptiveResponseObserver
from pure_visibility170 import window
from physical_publication import evaluate
from run_rotor_only import serial


def source(n):
    return ROOT.parent/f'output/{"regional212_20261108" if n=="11" else "fov227_20261108"}/recording{n}/regional_adaptive.jsonl'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',default='15');p.add_argument('--start',type=float,default=10.);p.add_argument('--end',type=float,default=16.01);a=p.parse_args()
    OUT=ROOT.parent/f'output/joint242_20261108/recording{a.recording}_{a.start:g}_{a.end:g}';OUT.mkdir(parents=True,exist_ok=False)
    dll=ROOT.parent/'output/helix_cpu232_20261108/build/Release/helix_reference.dll'
    bank=JointPhaseBank(dll);cache=ROOT.parent/f'output/{"motion181_20261107" if a.recording=="15" else "phase185_20261107"}/cache_recording{a.recording}'
    files=['joint_phase_bank.py','joint_ring_phase.py','angular_phase_candidates.py','angular_projection.py','tools/replay_joint236.py']
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    for n in files:
        d=OUT/'source_snapshot'/n;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes((ROOT/n).read_bytes())
    protocol=dict(recording=a.recording,source=str(source(a.recording)),source_sha256=hashes,
        author_dll=str(dll),GPU_used=False,NN_used=False,RAW_FPS_measured=False,
        labels_read_during_inference=False,qualification_thresholds_changed=False,
        observed_frequency_not_metric_thrust=True,discovery_geometry_from_past_only=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf-8')
    rows=[json.loads(s) for s in source(a.recording).read_text('utf-8').splitlines()]
    (OUT/'partial_periods.jsonl').touch()
    periods=[];timings=[];models={k:AdaptiveResponseObserver() for k in ['control','joint']};outputs={k:[] for k in models};delta=0.
    for idx,old in enumerate(rows):
        t=old['t'];tick=time.perf_counter()
        if a.start<=t<=a.end:
            u,q,meta=bank.step(t,window(cache,t),old['geometry'].get('rotors',[]),old['motion'])
        else:u=None;q=0.;meta=dict(observed_components={},learning_input_current=False,input_sigma_proxy=0.,outside_phase_pilot=True)
        periods.append(dict(t=t,u=u,quality=q,meta=meta));timings.append(1000*(time.perf_counter()-tick))
        with (OUT/'partial_periods.jsonl').open('a',encoding='utf-8') as fh:
            fh.write(json.dumps(periods[-1],default=serial,allow_nan=False)+'\n')
        for arm,m in models.items():
            r=copy.deepcopy(old)
            if arm=='control':
                pm=r['period_selection'];pu=r['force'].get('input_u');pu=None if pu is None else np.array(pu);quality=.7 if pu is not None else 0.
            else:pm=meta;pu=u;quality=q
            if r['state'].get('two_point_initialization_selected_by_partial_entry'):m.two_point_init=True
            c,v,f,s=m.step(t,r['body'],pu,quality,allow_gain_learning=pm.get('learning_input_current',True),input_meta=pm,motion=r['motion'],structure=r['structure'])
            r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'),period_selection=pm)
            r['physical']=evaluate(r,[.1,.2,.4]) if c else {};outputs[arm].append(r)
            if arm=='control' and c:
                for h,value in r['physical'].items():delta=max(delta,float(np.max(abs(np.array(value)-old['physical'][h]))))
        if idx%15==0:
            print(a.recording,idx,'qualified',meta.get('late_qualified_before_temporal_confirmation',0),'input',meta.get('qualified_current_rings',0),flush=True)
            (OUT/'progress.json').write_text(json.dumps(dict(last_frame=idx,t=t,phase_input_frames=sum(p['u'] is not None for p in periods),control_replay_delta=delta)), 'utf-8')
    assert delta<1e-7,delta
    for arm,rs in outputs.items():(OUT/f'{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf-8')
    (OUT/'periods.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in periods),'utf-8')
    assert 'torch' not in sys.modules
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    summary=dict(frames=len(rows),phase_pilot=[a.start,a.end],current_phase_input_frames=sum(r['u'] is not None for r in periods),
        late_qualified_frames=sum(p['meta'].get('late_qualified_before_temporal_confirmation',0)>0 for p in periods),
        known_ring_counts={str(k):sum(r['meta'].get('qualified_current_rings',0)==k for r in periods) for k in range(5)},
        CPU_phase_ms_mean=float(np.mean(timings)),CPU_phase_ms_p95=float(np.percentile(timings,95)),
        control_replay_max_px=delta,GPU_used=False,RAW_FPS_measured=False)
    (OUT/'completion.json').write_text(json.dumps(summary,indent=2),'utf-8');print(json.dumps(summary),flush=True)
