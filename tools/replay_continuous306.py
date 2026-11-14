"""Frozen305 inputs -> existing two-force observer and torque, same images."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from replay_joint236 import source
from adaptive_response_observer import AdaptiveResponseObserver
from continuous_phase_force import merge,torque_period
from rotor_torque_innovation import RotorTorqueInnovation,adapt
from physical_publication import evaluate
from score_rotor164 import score


if __name__=='__main__':
    OUT=ROOT.parent/'output/forecast306_20261109';OUT.mkdir(exist_ok=False)
    files=['continuous_phase_force.py','rotor_torque_innovation.py','adaptive_response_observer.py','tools/replay_continuous306.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(source_sha256=hashes,NN_used=False,GPU_used=False,RAW_FPS_measured=False,
        same_body_motion_and_geometry=True,no_recording_specific_rules=True,no_parameter_search=True,
        hidden_forces_not_filled=True,current_source_count_does_not_create_hidden_RPM=True,
        inference_completed_before_reference_reads=True,phase_inputs='frozen305',
        source_cutoffs='same25 queries10.8..15.6,0.4s',old_default_not_replaced=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    results={};counts={};parities={}
    for n in ('13','15','16'):
        src=source(n);rows=[json.loads(s) for s in src.read_text('utf8').splitlines()]
        periods={a:[json.loads(s) for s in (ROOT.parent/f'output/continuous305_20261109/recording{n}/{a}_periods.jsonl').read_text('utf8').splitlines()] for a in ('continuous','reset')}
        models={a:AdaptiveResponseObserver() for a in ('control','continuous_force','reset_force')}
        torque={a:RotorTorqueInnovation() for a in ('continuous_torque_only','continuous_force_torque')}
        outputs={a:[] for a in (*models,*torque)};delta=0.
        for i,old in enumerate(rows):
            t=old['t']
            for arm,model in models.items():
                r=copy.deepcopy(old)
                if arm=='control':
                    pm=r['period_selection'];u=r['force'].get('input_u');u=None if u is None else np.asarray(u,float);q=.7 if u is not None else 0.
                else:u,q,pm=merge(r,periods['continuous' if arm=='continuous_force' else 'reset'][i])
                if r['state'].get('two_point_initialization_selected_by_partial_entry'):model.two_point_init=True
                c,v,f,s=model.step(t,r['body'],u,q,allow_gain_learning=pm.get('learning_input_current',True),input_meta=pm,motion=r['motion'],structure=r['structure'])
                r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'),period_selection=pm)
                r['physical']=evaluate(r,[.1,.2,.4]) if c else {};outputs[arm].append(r)
                if arm=='control':
                    for h,p in r['physical'].items():delta=max(delta,float(np.max(abs(np.asarray(p)-old['physical'][h]))))
            for arm,model in torque.items():
                r=old if arm=='continuous_torque_only' else outputs['continuous_force'][-1]
                pp=torque_period(r,periods['continuous'][i]);info=model.observe(r,pp);s=adapt(r,info);s['physical']=evaluate(s,[.1,.2,.4]) if s['coefficient'] else {};outputs[arm].append(s)
        assert delta<1e-7,(n,delta);parities[n]=delta
        for arm,rr in outputs.items():
            (OUT/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rr),'utf8')
        counts[n]={a:dict(current_force_frames=sum(r['force']['active'] for r in rr),
            gain_ever_learned=sum(bool(r['force'].get('gain_ever_learned')) for r in rr),
            max_input_history=max(r['force'].get('qualified_input_history',0) for r in rr)) for a,rr in outputs.items()}
        print(n,'inference frozen; parity',delta,flush=True)
    for n in ('13','15','16'):
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        results[n]={a:score(OUT/f'{n}_{a}.jsonl',ref,origins) for a in outputs}
        print(n,{a:v['error'] for a,v in results[n].items()},flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(cases=results,counts=counts,default_replay_parity=parities,protocol=protocol),indent=2),'utf8')
