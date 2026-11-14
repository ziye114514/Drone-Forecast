"""One feasibility audit before adding differential thrust to predictions."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_spatial_moment import contrast,from_joint_period
from replay_joint236 import source
from axis_amplitude_bridge import current_axis_qualified
from propeller_physics import wrench


def controls():
    p=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]]);f=np.array([150.,220.,170.,210.]);x=contrast(p,f,radius_px=1.)
    y=contrast(p[::-1]+[90.,-60.],f[::-1],radius_px=1.)
    assert np.allclose(x['normal_acceleration_contrast_xy'],y['normal_acceleration_contrast_xy'],atol=1e-12)
    assert np.linalg.norm(contrast(p,np.ones(4)*200,radius_px=1.)['normal_acceleration_contrast_xy'])<1e-12
    c=contrast(p,np.ones(4)*200,radius_px=1.,coefficient_scales=[1.,1.2,1.,1.])
    assert np.linalg.norm(c['normal_acceleration_contrast_xy'])>0
    forces=np.array(x['individual_f2_weights']);F,T=wrench(np.c_[p,np.zeros(4)],forces)
    n=np.array([0.,0.,1.]);normal_ddot=np.cross(T,n)
    assert np.allclose(normal_ddot[:2],x['normal_acceleration_contrast_xy'])
    assert np.allclose(F[:2],0.)
    return dict(permutation_and_translation_invariant=True,equal_four_thrusts_zero_tilt_torque=True,
        unequal_coefficient_at_equal_RPM_changes_moment=True,
        differential_thrust_keeps_parallel_total_force_direction=True,
        projected_normal_acceleration_sign_matches_existing_wrench=True)


if __name__=='__main__':
    OUT=ROOT.parent/'output/torque294_20261109';OUT.mkdir(exist_ok=False)
    files=['rotor_spatial_moment.py','tools/audit_torque294.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(change='observed per-rotor f2 spatial contrast, not just sum',source_sha256=hashes,
        source_recordings=['13','15','16'],labels_read=False,NN_used=False,GPU_used=False,
        no_force_parameter_search=True,no_recording_specific_rules=True,hidden_torque_not_filled=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    results={};checks=controls()
    for n in protocol['source_recordings']:
        rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()]
        periods=[json.loads(s) for s in (ROOT.parent/f'output/joint242_20261108/recording{n}_10_16.01/periods.jsonl').read_text('utf8').splitlines()]
        assert len(rows)==len(periods)
        records=[];runs=[];run=[]
        for row,period in zip(rows,periods):
            assert row['t']==period['t'];m=from_joint_period(period);t=row['t'];r=dict(t=t,moment=m)
            if m['qualified']:
                nn=row['structure'].get('axis_xy');good=current_axis_qualified(row['structure']) and nn is not None
                if good:
                    nn=np.asarray(nn);nn=nn/np.linalg.norm(nn);tangent=np.array([-nn[1],nn[0]])
                    mvec=np.asarray(m['normal_acceleration_contrast_xy']);cov=np.array(m['weight_covariance_proxy'])
                    r.update(azimuth_drive=float(tangent@mvec),azimuth_drive_sigma_proxy=float(np.sqrt(max(0.,tangent@cov@tangent))),
                        radial_drive=float(nn@mvec),current_plane_qualified=True)
                else:r['current_plane_qualified']=False
                if run and (t-run[-1]['t']>.040001 or m['source_ids']!=run[-1]['moment']['source_ids']):runs.append(run);run=[]
                run.append(r)
            elif run:runs.append(run);run=[]
            records.append(r)
        if run:runs.append(run)
        summaries=[dict(start=r[0]['t'],end=r[-1]['t'],frames=len(r),sources=r[0]['moment']['source_ids'],
            contrast_variation=float(np.sqrt(np.mean(np.var([x['moment']['normal_acceleration_contrast_xy'] for x in r],axis=0))))) for r in runs]
        results[n]=dict(frames=len(rows),current_multi_rotor_frames=sum(r['moment']['qualified'] for r in records),
            longest_same_source_run_frames=max([len(r) for r in runs]+[0]),
            source_runs=summaries,records=records,
            actual_FDE_improvement_claimed=False)
        print(n,'multi rotor',results[n]['current_multi_rotor_frames'],'longest',results[n]['longest_same_source_run_frames'],flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(cases=results,checks=checks,protocol=protocol),indent=2),'utf8')
