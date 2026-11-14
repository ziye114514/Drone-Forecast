"""One frozen visibility-gauge torque trial; original image state unchanged."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_torque_partition import RotorTorquePartition
from rotor_torque_response import adapt
from replay_joint236 import source
from physical_publication import evaluate
from score_rotor164 import score


def mechanisms():
    def packet(t,values):
        return dict(t=t,meta=dict(records=[dict(qualified=True,current_force_admitted=True,track_id=k,proxy_twofold_hz=f,
            result=dict(arms=dict(joint_pose=dict(inferred_geometry=dict(cx=x,cy=0.,major=2.,minor=1.,angle=0.)))))
            for k,x,f in values]))
    def row(t):return dict(t=t,body=dict(detected=True,center=[0.,0.]),structure=dict(mode='four',geometry_epoch=0,
        current_geometric_center=[0.,0.],axis_xy=[0.,-1.],continuous=dict(updated=True)))
    b=RotorTorquePartition();a=b.observe(row(0),packet(0,[(0,-1.,200.),(1,1.,250.)]))
    c=b.observe(row(1/30),packet(1/30,[(0,-1.,200.)]))
    d=b.observe(row(2/30),packet(2/30,[(0,-1.,200.),(1,1.,300.)]))
    e=b.observe(row(3/30),packet(3/30,[(0,-1.,220.),(1,1.,300.)]))
    assert np.linalg.norm(c['known_torque_change_gauge_xy'])==0
    assert np.linalg.norm(d['known_torque_change_gauge_xy'])==0
    assert np.linalg.norm(e['known_torque_change_gauge_xy'])>0
    return dict(disappearance_adds_no_torque_impulse=True,newly_measured_changed_hidden_motor_adds_no_false_instantaneous_impulse=True,
        genuinely_continuing_source_frequency_change_retained=True,hidden_frequency_never_measured=True)


if __name__=='__main__':
    OUT=ROOT.parent/'output/partition297_20261109';OUT.mkdir(exist_ok=False)
    files=['rotor_torque_partition.py','rotor_torque_response.py','tools/pilot_partition297.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    protocol=dict(source_sha256=hashes,no_trajectory_labels_during_inference=True,position_state_frozen=True,
        source_membership_steps_are_unknown_response_transfers=True,hidden_motor_values_filled=False,
        no_recording_specific_rules=True,no_parameter_search=True,NN_used=False,GPU_used=False,RAW_FPS_measured=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    results={};checks=mechanisms()
    for n in ('13','15','16'):
        src=source(n);rows=[json.loads(s) for s in src.read_text('utf8').splitlines()]
        periods=[json.loads(s) for s in (ROOT.parent/f'output/joint242_20261108/recording{n}_10_16.01/periods.jsonl').read_text('utf8').splitlines()]
        bank=RotorTorquePartition();out=[];parity=0.
        for row,period in zip(rows,periods):
            assert row['t']==period['t'];info=bank.observe(row,period);new=adapt(row,info)
            if new['coefficient']:new['physical']=evaluate(new,[.1,.2,.4])
            out.append(new)
        path=OUT/f'{n}_partition.jsonl';path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in out),'utf8')
        infos=[r['rotor_torque_response'] for r in out]
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        results[n]=dict(frames=len(rows),current_period_frames=sum(i['current_source_count']>0 for i in infos),
            longest_current_block=max(i.get('current_block_frames',0) for i in infos),
            calibration_frames=sum(i.get('calibration_qualified',False) for i in infos),
            qualified_forecast_frames=sum(i['qualified'] for i in infos),
            best_gain_zscore=max(i.get('gain_zscore',0) for i in infos),
            reasons={r:sum(i['reason']==r for i in infos) for r in sorted(set(i['reason'] for i in infos))},
            control=score(src,ref,origins),partition=score(path,ref,origins))
        print(n,{k:v for k,v in results[n].items() if k not in ('control','partition')},flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(cases=results,checks=checks,protocol=protocol),indent=2),'utf8')
