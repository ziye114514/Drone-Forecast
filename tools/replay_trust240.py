"""Fixed-protocol trust in the geometry constraint; no GPU re-detection."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from geometry_trust_observer import GeometryTrustObserver
from physical_publication import evaluate
from score_rotor164 import score
from run_rotor_only import serial
from replay_joint236 import source


def replay(rows):
    m=GeometryTrustObserver();outputs=[]
    for old in rows:
        r=copy.deepcopy(old);pm=r['period_selection'];u=r['force'].get('input_u');u=None if u is None else np.asarray(u)
        if r['state'].get('two_point_initialization_selected_by_partial_entry'):m.two_point_init=True
        c,v,f,s=m.step(r['t'],r['body'],u,.7 if u is not None else 0.,allow_gain_learning=pm.get('learning_input_current',True),input_meta=pm,motion=r['motion'],structure=r['structure'])
        r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'))
        r['physical']=evaluate(r,[.1,.2,.4]) if c else {};outputs.append(r)
    return outputs


if __name__=='__main__':
    OUT=ROOT.parent/'output/trust241_20261108';OUT.mkdir(exist_ok=False)
    (OUT/'protocol.json').write_text(json.dumps(dict(cases=['13','15','16','11','side','train2'],
        change='causal trust of the geometry constraint on unknown force; same k*u and native position inputs',
        NN_used=False,GPU_used=False,labels_in_inference=False,recording_specific_settings=False,RAW_FPS_measured=False),indent=2),'utf-8')
    files=['geometry_trust_observer.py','tools/replay_trust240.py']
    for n in files:
        d=OUT/'source_snapshot'/n;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes((ROOT/n).read_bytes())
    results={}
    for n in ['13','15','16','11','side','train2']:
        if n in ('side','train2'):src=ROOT.parent/f'output/credit202_20261108/{n}/predictions.jsonl'
        else:src=source(n)
        rows=[json.loads(s) for s in src.read_text().splitlines()];outputs=replay(rows)
        if n=='15':
            prefix=replay(rows[:65]);assert all(a['physical']==b['physical'] for a,b in zip(prefix,outputs))
        path=OUT/f'{n}.jsonl';path.write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in outputs),'utf-8')
        if n in ('side','train2'):
            reference=ROOT.parent/f'output/{"dense_labels_side_110_1195" if n=="side" else "dense_labels_train2_40_46"}/labels.json'
            origin=np.arange(110.4,114.0001,.2) if n=='side' else np.arange(43.2,44.4001,.2)
            if not reference.exists():
                results[n]=dict(inference_complete=True,scoring_reference_pending=True);continue
        else:
            folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16','11':'regional212_20261108/reference11'}[n]
            reference=ROOT.parent/f'output/{folder}/references.json';origin=np.round(np.arange(10.8,15.6001,.2),1)
        results[n]=dict(control=score(src,reference,origin),trust=score(path,reference,origin))
        print(n,{k:v['error'].get('mean') for k,v in results[n].items()},flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(results=results,source_time_only=True,weak_outline_references_for_11_13_15_16=True),indent=2),'utf-8')
