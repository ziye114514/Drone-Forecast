"""Current local-source continuity + innovation-based unknown force.

NN observations remain frozen. No coordinate annotations are accessible.
All ablations use the same p/v and current motion measurements.
"""
from pathlib import Path
import sys,json,copy,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
from local_period_tracks import LocalPeriodTracks,InnovationUnknownForce
from visible_force_dynamics import VisibleForceDynamics,evaluate_visible
from run_force167 import pure_observation,pure_motion
from replay_gated_nn166 import guard
from run_rotor_only import serial

BASE=ROOT.parent/'output';PREV=BASE/'nn_rotor166_20261105';OUT=BASE/'source_force168_20261105/native_selection_fixed'
ARMS=('zero_algebra','legacy_algebra','tracked_algebra','zero_innovation','legacy_innovation','tracked_innovation','tracked_innovation_normal')
SOURCES=['local_period_tracks.py','visible_force_dynamics.py','rotor_thrust_bank.py','normal_turn_forecast.py',
    'flow_identifiable_gain.py','flow_force_balance.py','adaptive_force_memory.py','tools/run_source_force168.py','tools/run_force167.py']


def acceleration_covariance(motion):
    a=(motion or {}).get('acceleration') or {};se=a.get('standard_error_proxy')
    if a.get('applied') and se is not None:
        se=np.asarray(se,float)
        if se.shape==(2,) and np.isfinite(se).all():return np.diag(np.maximum(1.,se*se)),'current_acceleration_standard_error_proxy'
    return np.eye(2)*1e6,'existing1000px_s2_engineering_prior'


def replay(base_rows,observations,pure=False):
    track=LocalPeriodTracks();models={name:VisibleForceDynamics(learn=False) for name in ['legacy','tracked']}
    unknown={name:InnovationUnknownForce() for name in ['legacy','tracked','zero']}
    outputs={a:[] for a in ARMS};period_rows=[];previous=None;timings=[]
    for base,observation in zip(base_rows,observations):
        tick=time.perf_counter();t=base['t'];motion=pure_motion(base,previous) if pure else base.get('image_motion') or {}
        local=observation.get('local_temporal_phase') or {}
        continued=track.step(t,local)
        changed=copy.deepcopy(observation)
        if observation['structure'].get('mode') not in ('four','three','partial_two'):
            changed['pooled']=continued
        infos={name:model.step(base,observation if name=='legacy' else changed,motion) for name,model in models.items()}
        c=base.get('coefficient');net={};audit={};R,R_source=acceleration_covariance(motion)
        if c is not None:
            for name in ['legacy','tracked','zero']:
                i=infos['tracked' if name=='zero' else name]
                net[name],audit[name]=unknown[name].step(t,c['acceleration'],i['components'],i['normal'],200.,R,zero_input=name=='zero')
        for arm in ARMS:
            source='tracked' if arm.startswith('tracked') or arm=='zero_innovation' else 'legacy'
            info=copy.deepcopy(infos[source]);r=copy.deepcopy(base);r['coefficient']=copy.deepcopy(c)
            innovative='innovation' in arm
            if c is not None and innovative:
                name='zero' if arm=='zero_innovation' else source
                r['coefficient']['acceleration']=net[name].tolist()
                info.update(unknown_current=audit[name]['unknown'],current_net_acceleration=net[name].tolist())
            r.update(visible_force_dynamics=info,force168_arm=arm,structure=observation['structure'],
                local_source_tracking=continued,unknown_force_update=audit.get('zero' if arm=='zero_innovation' else source) if innovative else None,
                acceleration_covariance_source=R_source,
                force167_mode='zero' if arm.startswith('zero') else 'normal_only' if arm.endswith('_normal') else 'full_fixed')
            r['force']=dict(base['force'],active=info['current'],gain_learned=False,effective_gain_px_s2=200.,
                acceleration_px_s2=info['current_net_acceleration'],disturbance_px_s2=info['unknown_current'])
            r['physical']=evaluate_visible(r,[.1,.2,.4]);outputs[arm].append(r)
            if c is not None:assert r['coefficient']['position']==c['position'] and r['coefficient']['velocity']==c['velocity']
            assert r['valid']==base['valid']
        timings.append((time.perf_counter()-tick)*1000);previous=base
        period_rows.append(dict(t=t,original=observation.get('pooled'),continued=continued))
    summary=dict(frames=len(base_rows),current_raw_period=sum(bool((r.get('pooled') or {}).get('qualified')) for r in observations),
        current_tracked_period=sum(r['continued']['qualified'] for r in period_rows),
        source_epochs=len(set(r['continued']['source_epoch'] for r in period_rows)),
        continued_matches=sum(r['continued']['continued_regions'] for r in period_rows),
        all7arms_CPU_mean_ms=float(np.mean(timings)),all7arms_CPU_p95_ms=float(np.percentile(timings,95)),
        actual_RAW_FPS_measured=False,default_gain200_fixed=True,
        force_inputs={a:sum(r['visible_force_dynamics']['current'] for r in v) for a,v in outputs.items()})
    return outputs,summary,period_rows


if __name__=='__main__':
    sys.addaudithook(guard);OUT.mkdir(exist_ok=False);snapshot=OUT/'source_snapshot';snapshot.mkdir();hashes={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
    sources={};configs=json.loads((PREV/'summary.json').read_text('utf-8'))
    for clip,config in configs.items():
        obs=[json.loads(s) for s in (PREV/f'{clip}_observations.jsonl').read_text('utf-8').splitlines()]
        old={r['t']:r for r in map(json.loads,(Path(config['original_NN_run'])/'predictions.jsonl').read_text('utf-8').splitlines())}
        sources[clip]=([old[r['t']] for r in obs],obs,False,config['original_NN_run'])
    for clip,path in [('pure_side',BASE/'native_plane165_20261104/cold_relative_True.jsonl'),
        ('pure_train2',BASE/'native_plane165_20261104/train2_cache_border/predictions.jsonl')]:
        rows=[json.loads(s) for s in path.read_text('utf-8').splitlines()]
        obs=[dict(pure_observation(r),local_temporal_phase=r.get('local_temporal_phase')) for r in rows]
        sources[clip]=(rows,obs,True,str(path))
    summary={}
    for clip,(base,obs,pure,source) in sources.items():
        outputs,entry,periods=replay(base,obs,pure);prefix,_,_=replay(base[:len(base)//2],obs[:len(base)//2],pure)
        for arm,rows in outputs.items():
            assert [r['physical'] for r in prefix[arm]]==[r['physical'] for r in rows[:len(prefix[arm])]]
            (OUT/f'{clip}_{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rows),'utf-8')
        (OUT/f'{clip}_periods.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in periods),'utf-8')
        entry.update(NN_auxiliary_measurements_only=not pure,source=source,prefix_invariant=True)
        summary[clip]=entry;print(clip,json.dumps(entry),flush=True)
    summary['source_sha256']=hashes
    summary['source_changed_during_run']=[n for n,h in hashes.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
    assert not summary['source_changed_during_run']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
