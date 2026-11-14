"""Frozen two-factor visibility rule, matched ablations, same186 training teachers."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from orthogonal_visibility_response import samples,fit_model,SeparateVisibilityResponse
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    source=ROOT.parent/'output/training400_20261110';assert (source/'completion.json').exists();out=ROOT.parent/'output/orthogonal403_20261110';out.mkdir(exist_ok=False)
    files=['orthogonal_visibility_response.py','tools/evaluate_orthogonal403.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    groups=[];manifest={};completion=json.loads((source/'completion.json').read_text('utf8'))
    for n,status in completion['records'].items():
        if status['status']!='complete':continue
        rr=[json.loads(s) for s in (source/f'recording{n}/predictions.jsonl').read_text('utf8').splitlines()];g=samples(rr);groups.append(g);manifest[n]=len(g)
    model=fit_model(groups);(out/'shared_model.json').write_text(json.dumps(dict(model,training_manifest=manifest,all6_test_recordings_excluded_from_fit=True),indent=2),'utf8');print('one model frozen186 teachers',flush=True)
    arms=('none','no_geometry','no_count','constant');evaluation=('13','15','16','11','3','4')
    for n in evaluation:
        rr=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()]
        for a in arms:
            m=SeparateVisibilityResponse(model,ablation=a);vv=[m.step(r) for r in rr];assert all(r['valid']==old['valid'] and r['coefficient']==old['coefficient'] and r['force']==old['force'] for r,old in zip(vv,rr))
            (out/f'{n}_{a}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(n,'all inference frozen',flush=True)
    cases={};pool={a:[] for a in ('control','shared401','visibility402')+arms}
    for n in evaluation:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),shared401=score(ROOT.parent/f'output/training401_20261110/{n}.jsonl',ref,origins),visibility402=score(ROOT.parent/f'output/visibility402_20261110/{n}_visibility.jsonl',ref,origins),**{a:score(out/f'{n}_{a}.jsonl',ref,origins) for a in arms})
        for a,r in cases[n].items():pool[a].extend(q['error_px'] for q in r['records'] if q['valid'])
        print(n,{a:r['error'] for a,r in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),within35=sum(e<=35 for e in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,training_manifest=manifest,source_sha256=hashes,
        all6_evaluation_sources_excluded_from_fit=True,all_inference_before_reference_reads=True,
        one_model_no_recording_specific_selection=True,no_threshold_or_regularizer_scan=True,
        all_cases_exposed_development=True,RAW_FPS_certificate=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
