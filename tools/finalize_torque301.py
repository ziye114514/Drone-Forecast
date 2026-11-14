"""Freeze final incremental results and correct zero-input activity reporting."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_torque_innovation import RotorTorqueInnovation,adapt
from pilot_innovation299 import mechanisms
from physical_publication import evaluate
from score_rotor164 import score

if __name__=='__main__':
    OUT=ROOT.parent/'output/torque301_20261109';OUT.mkdir(exist_ok=False)
    assert (ROOT.parent/'output/temporal298_20261109/recording16/results.json').exists()
    checks=mechanisms();results={};parity={}
    files=['rotor_torque_innovation.py','native_torque_inputs.py','rotor_torque_partition.py','rotor_torque_response.py','physical_publication.py','tools/finalize_torque301.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    for n in ('13','15','16'):
        base=ROOT.parent/'output/factory300_20261109'
        rows=[json.loads(s) for s in (base/f'{n}_control.jsonl').read_text('utf8').splitlines()]
        factory=[json.loads(s) for s in (base/f'{n}_torque.jsonl').read_text('utf8').splitlines()]
        tangent=[json.loads(s) for s in (ROOT.parent/f'output/identifiable289_20261108/{n}_tangent.jsonl').read_text('utf8').splitlines()]
        joint=[json.loads(s) for s in (ROOT.parent/f'output/joint242_20261108/recording{n}_10_16.01/periods.jsonl').read_text('utf8').splitlines()]
        arms=['native','tangent_plus_native']
        temporal=None
        if n!='13':
            temporal=[json.loads(s) for s in (ROOT.parent/f'output/temporal298_20261109/recording{n}/periods.jsonl').read_text('utf8').splitlines()]
            arms.append('temporal_joint')
        for arm in arms:
            m=RotorTorqueInnovation();out=[];delta=0.
            for i,baseline in enumerate(rows):
                row=tangent[i] if arm=='tangent_plus_native' else baseline
                period=temporal[i] if arm=='temporal_joint' else baseline['period_selection']
                info=m.observe(row,period);r=adapt(row,info);r['physical']=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {};out.append(r)
                if arm=='native':
                    for h,p in r['physical'].items():delta=max(delta,float(np.max(abs(np.array(p)-factory[i]['physical'][h]))))
            if arm=='native':assert delta<1e-7,(n,delta);parity[n]=delta
            path=OUT/f'{n}_{arm}.jsonl';path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in out),'utf8')
            folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
            ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
            results.setdefault(n,{})[arm]=dict(score=score(path,ref,origins),
                actually_modified_frames=sum(r['rotor_torque_innovation'].get('normal_components_modified',0)>0 for r in out))
            print(n,arm,results[n][arm]['score']['error'],flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(cases=results,checks=checks,
        factory300_prediction_parity_after_zero_activity_reporting_fix=parity,
        protocol=dict(source_sha256=hashes,no_parameter_search=True,no_recording_specific_rules=True,
            NN_used=False,GPU_used=False,RAW_FPS_measured=False,
            cumulative_combination_is_replay_verified_not_new_RAW_FPS=True,
            temporal_joint_inputs_tested_only_on_15_16_no_generalization_claim=True)),indent=2),'utf8')
