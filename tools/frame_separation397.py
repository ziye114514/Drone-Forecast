"""Body-plane response + fixed-image external residual, one global rule.

Known/estimated axial force stays in the original rotating-plane dynamics.
Only the added lumped residual prior uses fixed image axes. This does NOT
replace native rotor detection with boxes/NN or assume a yaw/heading direction.
Compare both preserved geometry and slow phase geometry, never per-video pick.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,numpy as np,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from conditional_unknown_force import teachers,fit,ConditionalUnknownForce
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    out=ROOT.parent/'output/frames397_20261110';out.mkdir(exist_ok=False)
    paths={n:ROOT.parent/f'output/combined396_20261110/{n}_geometry.jsonl' for n in ('13','15','16','11','3','4')};data={n:[json.loads(s) for s in p.read_text('utf8').splitlines()] for n,p in paths.items()}
    files=['conditional_unknown_force.py','tools/frame_separation397.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    groups={n:teachers(rr,'camera') for n,rr in data.items()};models={}
    for n,rr in data.items():
        model=fit([g for k,g in groups.items() if k!=n],'camera');models[n]=model;pred=ConditionalUnknownForce(model);values=[pred.step(r) for r in rr]
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in values),'utf8');print(n,'inference frozen',flush=True)
    (out/'models.json').write_text(json.dumps(dict(LOO=models,shared_for_future=fit(list(groups.values()),'camera')),indent=2),'utf8')
    cases={};pool={a:[] for a in ('control','disk382','fixed_frame382','phase_fixed_frame')}
    for n in data:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),disk382=score(ROOT.parent/f'output/unknown382_20261110/{n}_disk.jsonl',ref,origins),fixed_frame382=score(ROOT.parent/f'output/unknown382_20261110/{n}_camera.jsonl',ref,origins),phase_fixed_frame=score(out/f'{n}.jsonl',ref,origins))
        for a,r in cases[n].items():pool[a].extend(q['error_px'] for q in r['records'] if q['valid'])
        print(n,{a:r['error'] for a,r in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),within35=sum(e<=35 for e in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,source_sha256=hashes,one_rule_all_sources=True,no_manual_training_labels=True,all_cases_exposed_development=True,no_RAW_FPS_claim=True),indent=2),'utf8');print(pooled,flush=True)
