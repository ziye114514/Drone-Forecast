"""Freeze a dimensionally derived torque prior; same-origin FDE ablation."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_torque_innovation import RotorTorqueInnovation,adapt,dynamics,update,rotation_kernel
from replay_joint236 import source
from physical_publication import evaluate
from score_rotor164 import score
from adaptive_force_memory import kernel


def mechanisms():
    # Coupled and control observers are exactly nested when source changes are
    # zero, even if the plane wobbles. This checks the claimed innovation,
    # rather than mistaking an extra geometric filter for a rotor benefit.
    x=np.array([0.,0.,0.,8.]);p=np.diag([.01,1.,10.,256.]);c=x[:3].copy();C=p[:3,:3].copy()
    for k in range(60):
        F,Q=dynamics(1/30,4);A,W=dynamics(1/30,3);x=F@x;p=F@p@F.T+Q;c=A@c;C=A@C@A.T+W
        angle=.05*np.sin(k/5);x,p=update(x,p,angle,.1);c,C=update(c,C,angle,.1)
        assert np.max(abs(x[:3]-c))<1e-12
    for h in (.1,.2,.4):assert abs(rotation_kernel(-1/.35,h,.3,0)-kernel(-1/.35+.3j,h))<1e-12
    return dict(zero_rotor_change_with_body_wobble_has_zero_increment=True,zero_angular_acceleration_matches_existing_kernel=True,
        inherited_sigma_and_inertia_are_unverified_priors=True)


if __name__=='__main__':
    OUT=ROOT.parent/'output/innovation299_20261109';OUT.mkdir(exist_ok=False)
    files=['rotor_torque_innovation.py','rotor_torque_partition.py','physical_publication.py','tools/pilot_innovation299.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    protocol=dict(source_sha256=hashes,no_reference_reads_during_inference=True,no_recording_specific_rules=True,
        no_parameter_search=True,prior_derived_from_existing_force_gain_and_image_geometry=True,
        known_or_hidden_true_motor_thrust_not_supplied=True,position_velocity_state_frozen=True,
        NN_used=False,GPU_used=False,RAW_FPS_measured=False,old_default_changed=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8');checks=mechanisms();results={}
    for n in ('13','15','16'):
        src=source(n);rows=[json.loads(s) for s in src.read_text('utf8').splitlines()]
        periods=[json.loads(s) for s in (ROOT.parent/f'output/joint242_20261108/recording{n}_10_16.01/periods.jsonl').read_text('utf8').splitlines()]
        model=RotorTorqueInnovation();out=[];control_delta=0.
        for row,period in zip(rows,periods):
            info=model.observe(row,period);r=adapt(row,info)
            if row['coefficient']:
                v=evaluate(row,[.1,.2,.4])
                for h,p in v.items():control_delta=max(control_delta,float(np.max(abs(np.array(p)-row['physical'][h]))))
                r['physical']=evaluate(r,[.1,.2,.4])
            out.append(r)
        assert control_delta<1e-7,control_delta
        path=OUT/f'{n}_innovation.jsonl';path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in out),'utf8')
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        infos=[r['rotor_torque_innovation'] for r in out]
        results[n]=dict(control=score(src,ref,origins),innovation=score(path,ref,origins),
            modified_frames=sum(i.get('normal_components_modified',0)>0 for i in infos),
            informative_drive_frames=sum(abs(i.get('drive_change',0))>1e-12 for i in infos),
            maximum_abs_delta_angular_acceleration=max(abs(i.get('delta_acceleration_rad_s2',0)) for i in infos),
            largest_prior_gain=max(i.get('prior',{}).get('gain',0) for i in infos),
            default_prediction_parity_px=control_delta)
        print(n,results[n]['control']['error'],results[n]['innovation']['error'],'modified',results[n]['modified_frames'],flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(cases=results,checks=checks,protocol=protocol),indent=2),'utf8')
