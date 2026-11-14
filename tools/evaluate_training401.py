"""One shared native-response model trained outside ALL six evaluation sources.

No candidate search: fixed-image residual frame, existing five coefficients,
same prior/teacher gates. Training capture400 must finish with source hashes
unchanged. The6 original recordings never enter model fit; all inference is
frozen before their weak evaluation references are opened.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from conditional_unknown_force import teachers,fit,ConditionalUnknownForce
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    source=ROOT.parent/'output/training400_20261110';assert (source/'completion.json').exists(),'Do not fit until the frozen RAW capture400 has completed'
    completion=json.loads((source/'completion.json').read_text('utf8'));protocol=json.loads((source/'protocol.json').read_text('utf8'))
    evaluation=('13','15','16','11','3','4');assert not set(evaluation).intersection(protocol['training_source_ids'])
    out=ROOT.parent/'output/training401_20261110';out.mkdir(exist_ok=False);groups=[];manifest={}
    files=['conditional_unknown_force.py','tools/evaluate_training401.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    for n,status in completion['records'].items():
        if status['status']!='complete':manifest[n]=dict(status=status['status'],teachers=0);continue
        p=source/f'recording{n}/predictions.jsonl';raw=p.read_bytes();rows=[json.loads(s) for s in raw.decode('utf8').splitlines()];g=teachers(rows,'camera');groups.append(g)
        manifest[n]=dict(status='complete',sha256=hashlib.sha256(raw).hexdigest(),frames=len(rows),teachers=len(g),
            actual_first_t=None if not rows else rows[0]['t'],actual_last_t=None if not rows else rows[-1]['t'],fixed_requested_interval=[10.,40.])
    if sum(map(len,groups))<10:
        (out/'insufficient_teachers.json').write_text(json.dumps(dict(training_manifest=manifest,fit_performed=False,reason='fewer than10 qualified native teachers'),indent=2),'utf8');raise SystemExit('Insufficient native training evidence; no model fitted')
    model=fit(groups,'camera');model.update(training_manifest=manifest,all_evaluation_sources_excluded=list(evaluation),
        experimental=True,metric_thrust_measured=False,all_test_cases_previously_exposed_to_development=True,
        training_uses_only_RAW_native_observations=True,prior_or_feature_hyperparameter_search=False)
    (out/'shared_model.json').write_text(json.dumps(model,indent=2),'utf8');print('shared model frozen,teachers',model['training_samples'],flush=True)
    for n in evaluation:
        rr=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()];m=ConditionalUnknownForce(model);values=[m.step(r) for r in rr]
        assert all(r['valid']==old['valid'] and r['coefficient']==old['coefficient'] and r['body']==old['body'] for r,old in zip(values,rr))
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in values),'utf8');print(n,'inference frozen',flush=True)
    cases={};pool={a:[] for a in ('control','fixed_frame382','shared_outside_all6')}
    for n in evaluation:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),fixed_frame382=score(ROOT.parent/f'output/unknown382_20261110/{n}_camera.jsonl',ref,origins),shared_outside_all6=score(out/f'{n}.jsonl',ref,origins))
        for a,r in cases[n].items():pool[a].extend(q['error_px'] for q in r['records'] if q['valid'])
        print(n,{a:r['error'] for a,r in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),within35=sum(e<=35 for e in v),within30=sum(e<=30 for e in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,training_manifest=manifest,source_sha256=hashes,
        one_shared_model_all_evaluation_sources=True,all6_evaluation_sources_excluded_from_training=True,
        all_inference_before_reference_reads=True,all_test_cases_exposed_development=True,
        weak_native_references_not_COM_groundtruth=True,RAW_FPS_certificate=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
