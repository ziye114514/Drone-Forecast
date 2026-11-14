"""Existing current-period continuity + SAME frozen186 visibility model.

No retraining on evaluation recordings, new gate, changed threshold or source
selection. Report original102 origins separately from additional coverage.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from visibility_response import VisibilityResponse
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    out=ROOT.parent/'output/continuity405_20261110';out.mkdir(exist_ok=False);source=ROOT.parent/'output/response388_20261110';assert (source/'completion.json').exists()
    model_path=ROOT/'models/native_response_visibility_v1.json';model=json.loads(model_path.read_text('utf8'))
    files=['visibility_response.py','conic_observability.py','tools/combine_continuity405.py','models/native_response_visibility_v1.json'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    evaluation=('13','15','16','11','3','4')
    for n in evaluation:
        rr=[json.loads(s) for s in (source/f'{n}.jsonl').read_text('utf8').splitlines()];m=VisibilityResponse(model);vv=[m.step(r) for r in rr]
        assert all(r['valid']==old['valid'] and r['coefficient']==old['coefficient'] and r['force']==old['force'] for r,old in zip(vv,rr))
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8');print(n,'inference frozen',flush=True)
    cases={};pool={a:[] for a in ('control','visibility402','continued')};extra={a:[] for a in pool}
    for n in evaluation:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),visibility402=score(ROOT.parent/f'output/visibility402_20261110/{n}_visibility.jsonl',ref,origins),continued=score(out/f'{n}.jsonl',ref,origins))
        original={r['t'] for r in cases[n]['control']['records'] if r['valid']}
        for a,r in cases[n].items():
            selected=[q['error_px'] for q in r['records'] if q['valid'] and q['t'] in original];pool[a].extend(selected);extra[a].extend(q['error_px'] for q in r['records'] if q['valid'] and q['t'] not in original)
            r['same_original_origins']=dict(n=len(selected),mean=float(np.mean(selected)),p95=float(np.percentile(selected,95)))
        print(n,{a:r['same_original_origins'] for a,r in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),within35=sum(e<=35 for e in v),extra_origins=len(extra[a]),extra_mean=None if not extra[a] else float(np.mean(extra[a]))) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,source_sha256=hashes,one_frozen_model_all_sources=True,all6_test_sources_excluded_from_fit=True,new_gates_or_model_search=False,no_RAW_FPS_certificate=True),indent=2),'utf8');print('POOLED',pooled,flush=True)
