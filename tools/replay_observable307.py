"""One unit-invariant force-gain optimization on the fixed306 inputs."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from observable_force_gain import ObservableGainObserver,excitation
from adaptive_response_observer import AdaptiveResponseObserver
from continuous_phase_force import merge,torque_period
from rotor_torque_innovation import RotorTorqueInnovation,adapt
from physical_publication import evaluate
from replay_joint236 import source
from score_rotor164 import score


def mechanisms():
    n=[]
    for i in range(60):
        t=i/30;axis=np.array([np.sin(t),-np.cos(t)]);u=.25*axis;sig=.01
        n.append((t,u,axis,sig))
    constant=excitation(n);assert not constant['qualified'],constant
    varied=[(t,u*(1+.3*np.sin(9*t)),axis,sigma) for t,u,axis,sigma in n]
    a=excitation(varied);b=excitation([(t,u*7,axis,sigma*7) for t,u,axis,sigma in varied]);assert a['qualified']
    assert abs(a['reliability']-b['reliability'])<1e-12
    shift=excitation([(t,u+[.2,-.4],axis,sigma) for t,u,axis,sigma in varied])
    assert abs(a['reliability']-shift['reliability'])<1e-12
    return dict(constant_amplitude_rotating_plane_has_zero_gain_credit=True,arbitrary_input_unit_scaling_invariant=True,
        constant_external_offset_eliminated=True,input_excitation_is_not_true_force_calibration=True)


if __name__=='__main__':
    OUT=ROOT.parent/'output/observable307_20261109';OUT.mkdir(exist_ok=False)
    files=['observable_force_gain.py','continuous_phase_force.py','tools/replay_observable307.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    protocol=dict(source_sha256=hashes,no_parameter_search=True,no_recording_specific_rules=True,
        same_frozen_phase_body_motion_and_geometry=True,NN_used=False,GPU_used=False,RAW_FPS_measured=False,
        input_gain_gate_changed_not_force_prior_amplitude=True,uncertainty_proxy_not_ground_truth=True,
        reference_reads_after_all_inference=True,baseline_kept=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8');checks=mechanisms();results={};diagnostics={}
    for n in ('13','15','16'):
        rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()]
        phases=[json.loads(s) for s in (ROOT.parent/f'output/continuous305_20261109/recording{n}/continuous_periods.jsonl').read_text('utf8').splitlines()]
        models=dict(control=AdaptiveResponseObserver(),observable=ObservableGainObserver());out={a:[] for a in models};torques={a:RotorTorqueInnovation() for a in models}
        for old,phase in zip(rows,phases):
            for arm,model in models.items():
                r=copy.deepcopy(old);u,q,pm=merge(r,phase)
                if r['state'].get('two_point_initialization_selected_by_partial_entry'):model.two_point_init=True
                c,v,f,s=model.step(r['t'],r['body'],u,q,allow_gain_learning=pm.get('learning_input_current',True),input_meta=pm,motion=r['motion'],structure=r['structure'])
                r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'),period_selection=pm)
                period=torque_period(r,phase);r=adapt(r,torques[arm].observe(r,period));r['physical']=evaluate(r,[.1,.2,.4]) if c else {};out[arm].append(r)
        for arm,rs in out.items():(OUT/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rs),'utf8')
        diagnostics[n]={a:dict(gain_learned_frames=sum(r['force'].get('gain_learned',False) for r in rr),
            largest_gain=max(r['force']['effective_gain_px_s2'] for r in rr),
            input_gate_qualified_frames=sum(r['force'].get('input_gain_qualification',{}).get('qualified',False) for r in rr)) for a,rr in out.items()}
        print(n,diagnostics[n],flush=True)
    for n in ('13','15','16'):
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        results[n]={a:score(OUT/f'{n}_{a}.jsonl',ref,origins) for a in models}
        print(n,{a:v['error'] for a,v in results[n].items()},flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(cases=results,diagnostics=diagnostics,checks=checks,protocol=protocol),indent=2),'utf8')
