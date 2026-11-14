"""One positive-gain uncertainty model, fixed ahead of all three scores."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_gain_mixture import RotorGainMixture
from rotor_frequency_drive import change
from continuous_phase_force import torque_period
from rotor_torque_innovation import RotorTorqueInnovation,adapt
from physical_publication import evaluate
from replay_joint236 import source
from score_rotor164 import score


def mechanisms():
    a={'a':dict(weight=.2,lever=np.array([1.,2.]),instance=1),'b':dict(weight=.3,lever=np.array([-1.,-2.]),instance=2)}
    b={k:dict(v,lever=v['lever']+[15.,-8.]) for k,v in a.items()}
    assert change(b,a,np.array([1.,0.]),1.)[0]==0
    b['a']['weight']=.25;assert change(b,a,np.array([1.,0.]),1.)[0]!=0
    b['a']['instance']=9;assert change(b,a,np.array([1.,0.]),1.)[0]==0
    return dict(geometry_only_motion_does_not_create_thrust_change=True,real_frequency_change_retained=True,
        reinitialized_phase_does_not_create_false_thrust_step=True,gain_bank_is_causal_prior_quadrature_not_FDE_parameter_search=True)


if __name__=='__main__':
    OUT=ROOT.parent/'output/gain308_20261109';OUT.mkdir(exist_ok=True)
    assert not (OUT/'results.json').exists(),'Completed evidence must not be overwritten'
    files=['rotor_gain_mixture.py','rotor_frequency_drive.py','rotor_torque_innovation.py','tools/replay_gain308.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    protocol=dict(source_sha256=hashes,NN_used=False,GPU_used=False,RAW_FPS_measured=False,
        no_recording_specific_rules=True,no_FDE_parameter_selection=True,scales_and_prior_identical_across_recordings=True,
        gain_posterior_updates_only_from_matured_planes=True,all_inference_before_reference_reads=True,
        frequency_only_input_difference=True,current_position_velocity_acceleration_frozen=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8');checks=mechanisms();results={};summary={}
    for n in ('13','15','16'):
        rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()]
        phases=[json.loads(s) for s in (ROOT.parent/f'output/continuous305_20261109/recording{n}/continuous_periods.jsonl').read_text('utf8').splitlines()]
        bank=RotorGainMixture();single=RotorTorqueInnovation(frequency_changes_only=True);out={a:[] for a in ('mixture','single_frequency_only')}
        for row,phase in zip(rows,phases):
            phase=torque_period(row,phase);r=bank.step(row,phase);s=adapt(row,single.observe(row,phase))
            for arm,value in [('mixture',r),('single_frequency_only',s)]:
                value['physical']=evaluate(value,[.1,.2,.4]) if value['coefficient'] else {};out[arm].append(value)
        for a,rr in out.items():(OUT/f'{n}_{a}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rr),'utf8')
        summary[n]=out['mixture'][-1]['rotor_mechanical_gain_mixture'];print(n,'last scale',summary[n]['posterior_mean_scale'],flush=True)
    for n in ('13','15','16'):
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        results[n]={a:score(OUT/f'{n}_{a}.jsonl',ref,origins) for a in out}
        print(n,{a:v['error'] for a,v in results[n].items()},flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(cases=results,last_gain_posterior=summary,checks=checks,protocol=protocol),indent=2),'utf8')
