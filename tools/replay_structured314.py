"""Exact four-source force-state reparameterization, fixed raw-derived inputs."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from structured_rotor_force import StructuredAdaptiveForce,StructuredNormalForce,collapse
from adaptive_response_observer import AdaptiveResponseObserver
from continuous_phase_force import merge,torque_period
from full_rotor_response_bridge import FullRotorResponseBridge
from physical_publication import evaluate
from replay_joint236 import source
from score_rotor164 import score


def mechanisms():
    rng=np.random.default_rng(303);x=rng.normal(size=14);x[6]=200;x[11]=1500
    A=rng.normal(size=(14,14));P=A@A.T;n=np.array([.2,-.9]);u=.6*n
    y,C,info=collapse(x,P,n,u);assert info['applied']
    H=np.zeros((2,14));H[:,4:6]=np.eye(2);H[:,6]=u
    E=np.zeros((2,14));E[:,4:6]=np.eye(2);E[:,11]=-n
    assert np.max(abs(H@x-H@y))<1e-10;assert np.max(abs(H@P@H.T-H@C@H.T))<1e-10
    assert np.max(abs(E@x-E@y))<1e-10;assert np.max(abs(E@P@E.T-E@C@E.T))<1e-10
    old=np.array([0.,-1.]);new=np.array([.3,-np.sqrt(1-.09)])
    m=StructuredNormalForce();m._initialize(np.zeros(2));m.normal=old;m.u=.3*old;m.x[6]=200;m.x[11]=1000;m.x[4:6]=np.array([50.,900.])+1000*old
    m.components={'phase_slot:0:0':.3*old};m.incoming_structure=dict(axis_xy=new.tolist())
    m.incoming_components={'phase_slot:0:1':.2*new}
    m._set_input(.2*new,transfer_delta=.2*new-.3*old,transfer_cov=np.zeros((2,2)))
    m.x[4:6]+=m.x[11]*(new-old);net=m.x[4:6]+m.x[6]*m.u;wanted=np.array([50.,900.])+(1000+60)*new
    assert np.max(abs(net-wanted))<1e-10,(net,wanted)
    return dict(full_source_change_preserves_current_net_acceleration_and_covariance=True,
        external_gravity_wind_residual_and_covariance_preserved=True,
        simultaneous_plane_and_visibility_change_does_not_change_total_normal_force=True,
        mechanical_CT_mass_or_inertia_not_measured=True)


if __name__=='__main__':
    OUT=ROOT.parent/'output/structured314_20261109';OUT.mkdir(exist_ok=False)
    files=['structured_rotor_force.py','continuous_phase_force.py','full_rotor_response_bridge.py','rotor_frequency_drive.py','tools/replay_structured314.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        d=OUT/'source_snapshot'/f;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes((ROOT/f).read_bytes())
    protocol=dict(source_sha256=hashes,NN_used=False,GPU_used=False,RAW_FPS_measured=False,
        phase_inputs='bootstrap311, automatic raw initialization only',no_parameter_search=True,no_recording_specific_rules=True,
        full_four_known_means_no_duplicate_fast_hidden_normal_response=True,
        coordinate_change_preserves_current_mean_and_covariance=True,all_inference_before_reference_reads=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8');checks=mechanisms();results={};diagnostics={}
    for n in ('13','15','16'):
        rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()]
        periods=[json.loads(s) for s in (ROOT.parent/f'output/bootstrap311_20261109/recording{n}/periods.jsonl').read_text('utf8').splitlines()]
        models=dict(control=AdaptiveResponseObserver(),structured=StructuredAdaptiveForce());out={k:[] for k in ('control','structured','structured_torque')};torque=FullRotorResponseBridge(redistribution_only=True)
        for old,phase in zip(rows,periods):
            for arm,m in models.items():
                r=copy.deepcopy(old);u,q,pm=merge(r,phase)
                if r['state'].get('two_point_initialization_selected_by_partial_entry'):m.two_point_init=True
                c,v,f,s=m.step(r['t'],r['body'],u,q,allow_gain_learning=pm.get('learning_input_current',True),input_meta=pm,motion=r['motion'],structure=r['structure'])
                r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'),period_selection=pm)
                r['physical']=evaluate(r,[.1,.2,.4]) if c else {};out[arm].append(r)
            r=out['structured'][-1];r=torque.step(r,torque_period(r,phase));r['physical']=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {};out['structured_torque'].append(r)
        for arm,rs in out.items():(OUT/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rs),'utf8')
        diagnostics[n]={arm:dict(anchor_seen_frames=sum(r['force'].get('structured_rotor_calibration',{}).get('full_anchor_seen',False) for r in rs),
            learned_gain_frames=sum(r['force'].get('gain_learned',False) for r in rs),
            max_gain=max(r['force']['effective_gain_px_s2'] for r in rs)) for arm,rs in out.items()}
        print(n,diagnostics[n],flush=True)
    for n in ('13','15','16'):
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        results[n]={a:score(OUT/f'{n}_{a}.jsonl',ref,origins) for a in out}
        print(n,{a:v['error'] for a,v in results[n].items()},flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(cases=results,diagnostics=diagnostics,checks=checks,protocol=protocol),indent=2),'utf8')
