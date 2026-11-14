"""Training-only gate for one nonlinear future unknown-rate contribution."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from quadratic_unknown_response import drag_rate,fit_model,QuadraticResponse
from conditional_unknown_force import TAU
from separated_force_response import rate_kernel
from extend_inverse321 import references
from score_rotor164 import score


def errors(model,g):
    out=[];theta=np.asarray(model['weights']);mu=float(model['visibility_context_mean']);weight=model['quadratic_unknown_rate_weight']
    for s in g:
        X=np.column_stack((s['X'],(s['visibility_weight']-mu)*np.asarray(s['X'])));out.append(float(np.linalg.norm(X@theta+weight*np.asarray(s['drag_feature'])-s['y'])))
    return out


if __name__=='__main__':
    source=ROOT.parent/'output/expanded411_412_20261110';assert (source/'results.json').exists()
    out=ROOT.parent/'output/quadratic415_20261110';out.mkdir(exist_ok=False);protocol=ROOT.parent/'output/checkpoints/quadratic415_predeclared_20261110.json';(out/'protocol.json').write_bytes(protocol.read_bytes())
    groups={}
    for n in ('1','2','5','6','7','8','9','10','12','14','17','18','19'):
        g=json.loads((source/f'teachers{n}.json').read_text('utf8'));lookup={}
        for batch in ('training400_20261110','training408_20261110'):
            for s in (ROOT.parent/f'output/{batch}/recording{n}/predictions.jsonl').read_text('utf8').splitlines():
                r=json.loads(s);lookup[r['t']]=r
        for s in g:s['drag_feature']=(rate_kernel(-1/TAU,0.,s['h'])*drag_rate(lookup[s['t']])).tolist()
        if g:groups[n]=g
    files=['tools/evaluate_quadratic415.py','quadratic_unknown_response.py','visibility_response.py','conditional_unknown_force.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    cv={}
    for arm,enabled in [('base',False),('quadratic',True)]:
        cv[arm]={};pooled=[]
        for n,g in groups.items():
            m=fit_model([gg for k,gg in groups.items() if k!=n],enabled);e=errors(m,g);pooled.extend(e);cv[arm][n]=dict(n=len(e),mean=float(np.mean(e)),drag_weight=m['quadratic_unknown_rate_weight'])
        cv[arm]['pooled']=dict(n=len(pooled),mean=float(np.mean(pooled)),median=float(np.median(pooled)));print('TRAIN CV',arm,cv[arm]['pooled'],flush=True)
    passed=cv['quadratic']['pooled']['mean']<cv['base']['pooled']['mean'];(out/'training_gate.json').write_text(json.dumps(dict(cv=cv,passed=passed,no_evaluation_data_used=True),indent=2),'utf8')
    if not passed:
        (out/'negative_result.json').write_text(json.dumps(dict(training_gate_failed=True,evaluation_inference_run=False,evaluation_references_opened=False,source_sha256=hashes),indent=2),'utf8');print('TRAINING GATE FAILED; NO TEST EVALUATION',flush=True);raise SystemExit(0)
    model=fit_model(list(groups.values()));model.update(all6_evaluation_sources_excluded_from_fit=True,training_manifest={n:len(g) for n,g in groups.items()})
    (out/'shared_model.json').write_text(json.dumps(model,indent=2),'utf8');evaluation=('13','15','16','11','3','4');print('MODEL FROZEN drag',model['quadratic_unknown_rate_weight'],flush=True)
    for n in evaluation:
        rr=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()]
        for arm,disabled in [('quadratic',False),('disabled',True)]:
            m=QuadraticResponse(model,disabled);vv=[m.step(r) for r in rr];assert all(r['valid']==old['valid'] and r['force']==old['force'] and r['coefficient']==old['coefficient'] for r,old in zip(vv,rr))
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(n,'inference frozen',flush=True)
    cases={};pool={a:[] for a in ('control','visibility402','quadratic','disabled')}
    for n in evaluation:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),visibility402=score(ROOT.parent/f'output/visibility402_20261110/{n}_visibility.jsonl',ref,origins),**{a:score(out/f'{n}_{a}.jsonl',ref,origins) for a in ('quadratic','disabled')})
        for a,r in cases[n].items():pool[a].extend(q['error_px'] for q in r['records'] if q['valid'])
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),within35=sum(e<=35 for e in v),within30=sum(e<=30 for e in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,source_sha256=hashes,all6_evaluation_sources_excluded_from_fit=True,training_only_gate=True,all_inference_before_reference_reads=True,independent_new_holdout_claimed=False,RAW_FPS_certificate=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
