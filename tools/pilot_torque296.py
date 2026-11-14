"""One causal differential-thrust response trial, original position state frozen."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_spatial_moment import from_joint_period,from_slot_period
from rotor_torque_response import RotorTorqueResponse,integrated_drive,fit,adapt
from rotor_thrust_bank import RotorThrustBankProxy
from replay_joint236 import source
from physical_publication import evaluate
from score_rotor164 import score


def mechanisms():
    t=np.arange(61)/30;u=.2*np.sin(9*t)+.05*np.cos(4*t);_,integral=integrated_drive(t,u)
    angle=.3+.2*t+.5*1.7*t*t+12*integral
    block=list(zip(t,angle,u,np.ones(len(t))*.01));f=fit([block]);assert f['qualified'],f
    assert abs(f['effective_azimuth_response_gain']-12)<1e-9,f
    shifted=list(zip(t,angle+4.7-.8*t+.5*9*t*t,u,np.ones(len(t))*.01))
    g=fit([shifted]);assert abs(f['effective_azimuth_response_gain']-g['effective_azimuth_response_gain'])<1e-9
    constant=list(zip(t,angle,np.ones(len(t))*.2,np.ones(len(t))*.01))
    c=fit([constant]);assert c['data_only_gain_information']==0,c
    return dict(synthetic_response_gain=f['effective_azimuth_response_gain'],unknown_constant_torque_and_initial_motion_invariant=True,
        constant_unequal_thrust_is_not_identifiable_under_unknown_balancing_torque=True,real_rotor_calibration_not_claimed=True)


if __name__=='__main__':
    OUT=ROOT.parent/'output/torque296_20261109';OUT.mkdir(exist_ok=False)
    files=['rotor_spatial_moment.py','rotor_torque_response.py','tools/pilot_torque296.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    protocol=dict(source_sha256=hashes,position_velocity_acceleration_frozen=True,trajectory_labels_read_only_after_inference=True,
        no_recording_specific_rules=True,no_force_gain_parameter_search=True,hidden_forces_or_torques_not_filled=True,
        GPU_used=False,NN_used=False,RAW_FPS_measured=False,old_default_changed=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    checks=mechanisms();scores={}
    for n in ('13','15','16','clear'):
        if n=='clear':
            src=ROOT.parent/'output/motion_consistency_30px_20261028/RAW_clear_release_candidate2s/predictions.jsonl'
            periods=None
        else:
            src=source(n);periods=[json.loads(s) for s in (ROOT.parent/f'output/joint242_20261108/recording{n}_10_16.01/periods.jsonl').read_text('utf8').splitlines()]
        rows=[json.loads(s) for s in src.read_text('utf8').splitlines()];learner=RotorTorqueResponse();bank=RotorThrustBankProxy();out=[];moments=[]
        for j,row in enumerate(rows):
            if periods is None:
                _,_,pm=bank.step(row['t'],row['structure'],row['geometry'].get('rotors',[]),row['rates'],{})
                moment=from_slot_period(row,pm)
            else:moment=from_joint_period(periods[j])
            info=learner.observe(row['t'],row['structure'],moment);new=adapt(row,info)
            if new['coefficient']:new['physical']=evaluate(new,[.1,.2,.4])
            out.append(new);moments.append(dict(t=row['t'],moment=moment,learning=info))
        path=OUT/f'{n}_torque.jsonl';path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in out),'utf8')
        (OUT/f'{n}_moments.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in moments),'utf8')
        scores[n]=dict(frames=len(rows),multi_rotor_frames=sum(r['moment']['qualified'] for r in moments),
            calibration_qualified_frames=sum(r['learning'].get('calibration_qualified',False) for r in moments),
            torque_forecast_qualified_frames=sum(r['learning']['qualified'] for r in moments),
            reasons={reason:sum(r['learning']['reason']==reason for r in moments) for reason in sorted(set(r['learning']['reason'] for r in moments))},
            best_gain_zscore=max([r['learning'].get('gain_zscore',0) for r in moments]),
            longest_current_source_block=max(r['learning'].get('current_block_frames',0) for r in moments))
        if n!='clear':
            folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
            ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
            scores[n].update(control=score(src,ref,origins),torque=score(path,ref,origins))
        else:scores[n]['no_continuous_future_reference']=True
        print(n,{k:v for k,v in scores[n].items() if k not in ('control','torque')},flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(cases=scores,checks=checks,protocol=protocol),indent=2),'utf8')
