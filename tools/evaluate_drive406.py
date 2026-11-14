"""Predeclared nonnegative collective-drive multiplier, fixed186/102 protocol."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from future_drive_response import drive_samples,fit_model,FutureDriveResponse
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    source=ROOT.parent/'output/training400_20261110';completion=json.loads((source/'completion.json').read_text('utf8'))
    protocol=ROOT.parent/'output/checkpoints/drive406_predeclared_20261110.json';assert protocol.exists()
    out=ROOT.parent/'output/drive406_20261110';out.mkdir(exist_ok=False);groups=[];manifest={}
    files=['future_drive_response.py','conditional_unknown_force.py','physical_publication.py','visibility_response.py','tools/evaluate_drive406.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    (out/'protocol.json').write_bytes(protocol.read_bytes())
    for n,status in completion['records'].items():
        assert n not in ('3','4','11','13','15','16')
        if status['status']!='complete':continue
        rr=[json.loads(s) for s in (source/f'recording{n}/predictions.jsonl').read_text('utf8').splitlines()];g=drive_samples(rr);groups.append(g);manifest[n]=len(g)
        (out/f'teachers{n}.json').write_text(json.dumps(g),'utf8');print(n,'teachers',len(g),flush=True)
    model=fit_model(groups);(out/'shared_model.json').write_text(json.dumps(dict(model,training_manifest=manifest,all6_evaluation_sources_excluded_from_fit=True),indent=2),'utf8');print('gamma',model['future_drive_multiplier'],flush=True)
    evaluation=('13','15','16','11','3','4')
    for n in evaluation:
        rr=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()]
        for arm,fixed in [('drive',None),('fixed_gamma',1.)]:
            pred=FutureDriveResponse(model,fixed_gamma=fixed);vv=[pred.step(r) for r in rr]
            assert all(r['valid']==old['valid'] and r['coefficient']==old['coefficient'] and r['force']==old['force'] for r,old in zip(vv,rr))
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(n,'inference frozen',flush=True)
    cases={};pool={a:[] for a in ('control','visibility402','drive','fixed_gamma')}
    for n in evaluation:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),visibility402=score(ROOT.parent/f'output/visibility402_20261110/{n}_visibility.jsonl',ref,origins),**{a:score(out/f'{n}_{a}.jsonl',ref,origins) for a in ('drive','fixed_gamma')})
        for a,r in cases[n].items():pool[a].extend(q['error_px'] for q in r['records'] if q['valid'])
        print(n,{a:r['error']['mean'] for a,r in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),within35=sum(e<=35 for e in v),within30=sum(e<=30 for e in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,training_manifest=manifest,source_sha256=hashes,
        gamma=model['future_drive_multiplier'],all6_evaluation_sources_excluded_from_fit=True,all_inference_before_reference_reads=True,
        one_global_model_no_recording_selection=True,no_hyperparameter_scan=True,independent_new_holdout_claimed=False,RAW_FPS_certificate=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
