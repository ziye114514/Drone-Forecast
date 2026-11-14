"""Optimize observed-rotor dynamics, with frozen NN measurement assistance."""
from pathlib import Path
import sys,json,copy,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
from visible_force_dynamics import VisibleForceDynamics,evaluate_visible
from run_rotor_only import serial
from replay_gated_nn166 import guard
from axis_amplitude_bridge import current_axis_qualified

BASE=ROOT.parent/'output';PREV=BASE/'nn_rotor166_20261105';OUT=BASE/'visible_force167_20261105/source_gain'
MODES=('zero','normal_only','full_fixed','full_learned')
SOURCES=('visible_force_dynamics.py','tools/run_force167.py','flow_identifiable_gain.py',
    'flow_force_balance.py','normal_turn_forecast.py','rotor_thrust_bank.py','adaptive_force_memory.py')
# Sources moved out of the repository root; resolve a flat name to wherever it lives now.
SOURCE_DIRS=(ROOT/'core',ROOT/'tools',ROOT)
def source_path(name):
    for base in SOURCE_DIRS:
        p=base/name
        if p.exists():return p
    raise FileNotFoundError('source not found: '+name)


def pure_observation(row):
    period=row.get('period') or {};local=row.get('local_temporal_phase') or {};meta=row.get('period_selection') or {}
    # Saved period.frequency_hz is the global pixel median even when the live
    # bank actually selected local first-harmonic evidence. Read the matching
    # recorded source, rather than label the global median as a local reading.
    use_selected='selected_frequency_hz' in period
    use_local=meta.get('pooled_estimator_source')=='local_pixel_first_harmonic' and local.get('qualified')
    frequency=period['selected_frequency_hz'] if use_selected else local.get('frequency_hz') if use_local else period.get('frequency_hz')
    pooled=dict(qualified=bool(frequency is not None),frequency_hz=frequency,
        selection_source=meta.get('pooled_estimator_source'),
        saved_frequency_field='period.selected_frequency_hz' if use_selected else 'local_temporal_phase.frequency_hz' if use_local else 'period.frequency_hz')
    s=row['structure'];g=row.get('geometry') or {}
    return dict(t=row['t'],structure=s,geometry=g,rates=row.get('rates',[]),pooled=pooled,current_pose=current_axis_qualified(s))


def pure_motion(row,previous=None):
    m=row.get('motion') or {};dt=m.get('dt_s');estimate=m.get('estimate') or {};obs={}
    inferred=False
    if dt is None and previous is not None and m.get('current_observation') and (previous.get('motion') or {}).get('current_observation') and 'dx' in estimate:
        a=row.get('body') or {};b=previous.get('body') or {}
        if a.get('detected') and b.get('detected') and np.allclose(np.asarray(a['center'])-b['center'],[estimate['dx'],estimate['dy']],atol=1e-6,rtol=0.):
            dt=row['t']-previous['t'];inferred=True
    if m.get('current_observation') and dt is not None and 0<dt<=.12 and 'dx' in estimate:
        # Reuse the actual observer's declared measurement prior: the stage165
        # relative observer uses2px displacement, not two6px absolute fixes.
        relative=bool((row.get('state') or {}).get('native_relative_displacement_observed'))
        R=np.eye(2)*((4. if relative else 2*36)/dt**2)
        obs=dict(valid=True,derivative_eligible=not m.get('support_reference_changed',False),
            midpoint_t=row['t']-dt/2,velocity=[estimate['dx']/dt,estimate['dy']/dt],
            velocity_covariance=R.tolist(),dt_recovered_from_exact_consecutive_reference=inferred)
    return dict(observation=obs,acceleration=dict(applied=False))


def replay(base_rows,observations,pure=False):
    models={m:VisibleForceDynamics(learn=m=='full_learned',gain_window_s=3. if m=='full_learned' else .35) for m in MODES};outputs={m:[] for m in MODES};cost={m:[] for m in MODES}
    previous=None
    for base,obs in zip(base_rows,observations):
        motion=pure_motion(base,previous) if pure else base.get('image_motion') or {}
        for mode,model in models.items():
            tick=time.perf_counter();r=copy.deepcopy(base);info=model.step(base,obs,motion)
            r.update(visible_force_dynamics=info,force167_mode=mode,structure=obs['structure'])
            r['force']=dict(base['force'],active=info['current'],effective_gain_px_s2=info['effective_gain'],
                gain_learned=info['gain_updated'],disturbance_px_s2=info['unknown_current'],
                acceleration_px_s2=info['current_net_acceleration'])
            r['physical']=evaluate_visible(r,[.1,.2,.4]);cost[mode].append((time.perf_counter()-tick)*1000)
            outputs[mode].append(r)
        previous=base
    summary={m:dict(frames=len(v),current=sum(r['visible_force_dynamics']['current'] for r in v),
        nonzero_correction=int(sum(np.linalg.norm(np.asarray(r['physical'].get('0.4',[0,0]))-np.asarray(outputs['zero'][i]['physical'].get('0.4',[0,0])))>1e-8 for i,r in enumerate(v))),
        gain_updates=models[m].updated_total,final_gain=models[m].gain,gain_fit_reasons=dict(models[m].reasons),
        force_mean_ms=float(np.mean(cost[m])),force_p95_ms=float(np.percentile(cost[m],95))) for m,v in outputs.items()}
    return outputs,summary


if __name__=='__main__':
    sys.addaudithook(guard);OUT.mkdir(exist_ok=False);snap=OUT/'source_snapshot';snap.mkdir();hashes={}
    for name in SOURCES:
        data=(source_path(name)).read_bytes();dest=snap/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
    configs=json.loads((PREV/'summary.json').read_text('utf-8'));summary={}
    for clip,config in configs.items():
        observations=[json.loads(s) for s in (PREV/f'{clip}_observations.jsonl').read_text('utf-8').splitlines()]
        old={r['t']:r for r in map(json.loads,(Path(config['original_NN_run'])/'predictions.jsonl').read_text('utf-8').splitlines())}
        base=[old[r['t']] for r in observations];outputs,entry=replay(base,observations)
        prefix,_=replay(base[:len(base)//2],observations[:len(base)//2])
        for mode,rows in outputs.items():
            assert [r['physical'] for r in prefix[mode]]==[r['physical'] for r in rows[:len(prefix[mode])]]
            assert [r['coefficient'] for r in rows]==[r['coefficient'] for r in base]
            (OUT/f'{clip}_{mode}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rows),'utf-8')
        summary[clip]=dict(arms=entry,NN_frozen_measurements_only=True,NN_forecasts_used=False,prefix_invariant=True)
        print(clip,json.dumps(entry),flush=True)
    for clip,path in [('pure_side',BASE/'side_temporal164_20261101/raw_side_final/predictions.jsonl'),
        ('pure_train2',BASE/'native_plane165_20261104/train2_cache_border/predictions.jsonl')]:
        base=[json.loads(s) for s in path.read_text('utf-8').splitlines()];obs=[pure_observation(r) for r in base]
        outputs,entry=replay(base,obs,True)
        for mode,rows in outputs.items():(OUT/f'{clip}_{mode}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rows),'utf-8')
        summary[clip]=dict(arms=entry,NN_used=False,source=str(path));print(clip,json.dumps(entry),flush=True)
    summary['source_sha256']=hashes
    summary['source_changed_during_run']=[n for n,h in hashes.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
    assert not summary['source_changed_during_run']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
