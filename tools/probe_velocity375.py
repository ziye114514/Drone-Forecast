"""Held-source-out physical innovation readout; fixed protocol and all cases."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from velocity_force_calibration import teachers,fit
from calibrated_force_rollout import CalibratedRollout
from extend_inverse321 import references
from score_rotor164 import score


if __name__=='__main__':
    out=ROOT.parent/'output/velocity375_20261109';out.mkdir(exist_ok=False)
    files=['velocity_force_calibration.py','calibrated_force_rollout.py','physical_publication.py','tools/probe_velocity375.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    data={n:[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()] for n in ('13','15','16','11','3','4')}
    own={n:teachers(rr) for n,rr in data.items()};camera={n:teachers(rr,True) for n,rr in data.items()};models={};arms=('full','camera_axis','no_future_plane')
    for n,rr in data.items():
        models[n]={}
        for arm in arms:
            groups=camera if arm=='camera_axis' else own;model=fit([g for k,g in groups.items() if k!=n],arm)
            models[n][arm]=model;predictor=CalibratedRollout(model);outputs=[predictor.step(r) for r in rr]
            assert all(r['valid']==old['valid'] and r['body']==old['body'] for r,old in zip(outputs,rr))
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in outputs),'utf8')
        print(n,'frozen', {a:models[n][a]['coefficients'] for a in arms},flush=True)
    (out/'models.json').write_text(json.dumps(dict(leave_one_recording_out=models,shared_model_for_future=fit(list(own.values()))),indent=2),'utf8')
    cases={}
    for n in data:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),**{a:score(out/f'{n}_{a}.jsonl',ref,origins) for a in arms});print(n,{a:v['error'] for a,v in cases[n].items()},flush=True)
    (out/'results.json').write_text(json.dumps(dict(cases=cases,source_sha256=hashes,manual_references_not_used_for_training=True,heldout_source_not_used_for_its_model=True,all_inference_before_reference_reads=True,all_cases_previously_exposed_development=True,NN_used=False,RAW_FPS_measured=False),indent=2),'utf8')
