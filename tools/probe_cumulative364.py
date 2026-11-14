"""One cumulative force candidate, identical order/rules for all recordings."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from integral_normal_response import IntegralNormalResponse
from anisotropic_response import AnisotropicResponse
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score


if __name__=='__main__':
    out=ROOT.parent/'output/cumulative364_20261109';out.mkdir(exist_ok=False)
    files=['integral_normal_response.py','anisotropic_response.py','tools/probe_cumulative364.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    for n in ('13','15','16','11','3','4'):
        rr=[json.loads(s) for s in (ROOT.parent/f'output/inverse326_20261109/{n}_inverse320.jsonl').read_text('utf8').splitlines()];integral=IntegralNormalResponse();disk=AnisotropicResponse();outputs=[]
        for old in rr:
            r=disk.step(integral.step(old));r['physical']=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {};assert r['coefficient']==old['coefficient'];outputs.append(r)
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in outputs),'utf8');print(n,'frozen',flush=True)
    cases={}
    for n in ('13','15','16','11','3','4'):
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/inverse326_20261109/{n}_inverse320.jsonl',ref,origins),candidate=score(out/f'{n}.jsonl',ref,origins));print(n,{k:v['error'] for k,v in cases[n].items()},flush=True)
    (out/'results.json').write_text(json.dumps(dict(cases=cases,source_sha256=hashes,current_pva_unchanged=True,all_cases_development=True,one_global_rule=True,no_parameter_search=True,RAW_FPS_measured=False,NN_used=False),indent=2),'utf8')
