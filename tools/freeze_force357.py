"""Canonical per-frame force-candidate outputs with fresh forecast fields."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from anisotropic_response import AnisotropicResponse
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score


if __name__=='__main__':
    out=ROOT.parent/'output/force357_20261109';out.mkdir(exist_ok=False)
    files=['anisotropic_response.py','physical_publication.py','tools/freeze_force357.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    parity={};inputs={}
    for n in ('13','15','16','11','3','4'):
        path=ROOT.parent/f'output/inverse326_20261109/{n}_inverse320.jsonl';inputs[n]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows=[json.loads(s) for s in path.read_text('utf8').splitlines()];old=[json.loads(s) for s in (ROOT.parent/f'output/anisotropic354_20261109/{n}.jsonl').read_text('utf8').splitlines()]
        m=AnisotropicResponse();outputs=[];delta=0.
        for r,expected in zip(rows,old):
            new=m.step(r);new['physical']=evaluate(new,[.1,.2,.4]) if new['coefficient'] else {}
            wanted=evaluate(expected,[.1,.2,.4]) if expected['coefficient'] else {}
            if wanted:delta=max(delta,max(float(np.max(abs(np.array(new['physical'][h])-v))) for h,v in wanted.items()))
            assert new['coefficient']==r['coefficient'];outputs.append(new)
        assert delta<1e-7,(n,delta);parity[n]=delta
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in outputs),'utf8');print(n,'frozen',delta,flush=True)
    cases={}
    for n in inputs:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/inverse326_20261109/{n}_inverse320.jsonl',ref,origins),candidate=score(out/f'{n}.jsonl',ref,origins))
    result=dict(cases=cases,source_sha256=hashes,input_sha256=inputs,parameter_reconstruction_same_evaluator_parity=parity,
        all_saved_physical_fields_recomputed=True,NN_used=False,no_parameter_search=True,RAW_FPS_measured=False,
        weak_visible_outline_references_not_independent_COM_truth=True)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8')
