"""Predeclared visibility response extension; train only400, fixed6 test sources."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from visibility_response import samples,fit_model,VisibilityResponse
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    source=ROOT.parent/'output/training400_20261110';assert (source/'completion.json').exists(),'Wait for all frozen training capture400'
    assert (ROOT.parent/'output/training401_20261110/results.json').exists(),'Run predeclared5-coefficient shared401 before visibility extension'
    out=ROOT.parent/'output/visibility402_20261110';out.mkdir(exist_ok=False);completion=json.loads((source/'completion.json').read_text('utf8'));groups=[];manifest={}
    files=['visibility_response.py','tools/evaluate_visibility402.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    for n,status in completion['records'].items():
        if status['status']!='complete':continue
        rr=[json.loads(s) for s in (source/f'recording{n}/predictions.jsonl').read_text('utf8').splitlines()];g=samples(rr);groups.append(g);manifest[n]=dict(teachers=len(g),visibility_mean=None if not g else float(np.mean([s['visibility_weight'] for s in g])))
    model=fit_model(groups);(out/'shared_model.json').write_text(json.dumps(dict(model,training_manifest=manifest,all6_evaluation_sources_excluded=True),indent=2),'utf8')
    evaluation=('13','15','16','11','3','4')
    for n in evaluation:
        rr=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()]
        for arm,constant in [('visibility',False),('constant_context',True)]:
            pred=VisibilityResponse(model,constant_context=constant);values=[pred.step(r) for r in rr];assert all(r['valid']==old['valid'] and r['coefficient']==old['coefficient'] and r['force']==old['force'] for r,old in zip(values,rr))
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in values),'utf8')
        print(n,'inference frozen',flush=True)
    cases={};pool={a:[] for a in ('control','fixed_frame382','shared401','visibility','constant_context')}
    for n in evaluation:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),fixed_frame382=score(ROOT.parent/f'output/unknown382_20261110/{n}_camera.jsonl',ref,origins),shared401=score(ROOT.parent/f'output/training401_20261110/{n}.jsonl',ref,origins),**{a:score(out/f'{n}_{a}.jsonl',ref,origins) for a in ('visibility','constant_context')})
        for a,r in cases[n].items():pool[a].extend(q['error_px'] for q in r['records'] if q['valid'])
        print(n,{a:r['error'] for a,r in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),within35=sum(e<=35 for e in v),within30=sum(e<=30 for e in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,training_manifest=manifest,source_sha256=hashes,
        all6_evaluation_sources_excluded_from_model_fit=True,all_inference_before_reference_reads=True,
        model_has10_scalars_not_an_NN=True,no_hyperparameter_or_recording_specific_search=True,
        independent_new_holdout_claimed=False,RAW_FPS_certificate=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
