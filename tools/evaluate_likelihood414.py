"""Training-only source-held-out choice of correlated-teacher likelihood."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from conditional_unknown_force import fit
from visibility_response import VisibilityResponse
from extend_inverse321 import references
from score_rotor164 import score


def selected(group,recipe):
    if recipe=='source_normalized':return group
    keep=[];last=None
    for s in sorted(group,key=lambda s:s['t']):
        if last is None or s['t']-last>=.4-1e-5:keep.append(s);last=s['t']
    return keep


def fitted(groups,recipe):
    used=[selected(g,recipe) for g in groups if g];mean=float(np.mean([np.mean([s['visibility_weight'] for s in g]) for g in used]));augmented=[]
    for g in used:augmented.append([dict(s,X=np.column_stack((s['X'],(s['visibility_weight']-mean)*np.asarray(s['X']))).tolist()) for s in g])
    model=fit(augmented,'camera',prior=np.zeros(10),group_normalize=recipe=='source_normalized')
    model.update(visibility_conditioned=True,visibility_context_mean=mean,one_global_model_all_sources=True,
        current_p_v_a_preserved=True,explicit_hidden_rotor_force_owner_unchanged=True,
        training_only_likelihood_selection=True,likelihood_recipe=recipe,all6_evaluation_sources_excluded_from_fit=True)
    return model


def errors(model,held):
    z=np.asarray(model['weights']);center=float(model['visibility_context_mean']);out=[]
    for s in held:
        X=np.column_stack((s['X'],(s['visibility_weight']-center)*np.asarray(s['X'])))
        out.append(float(np.linalg.norm(X@z-np.asarray(s['y']))))
    return out


if __name__=='__main__':
    source=ROOT.parent/'output/expanded411_412_20261110';assert (source/'results.json').exists(),'Wait for all frozen expanded teachers'
    # Do NOT read its evaluation results; only training teachers are inputs.
    out=ROOT.parent/'output/likelihood414_20261110';out.mkdir(exist_ok=False);protocol=ROOT.parent/'output/checkpoints/likelihood414_predeclared_20261110.json';(out/'protocol.json').write_bytes(protocol.read_bytes())
    groups={n:json.loads((source/f'teachers{n}.json').read_text('utf8')) for n in ('1','2','5','6','7','8','9','10','12','14','17','18','19')}
    groups={n:g for n,g in groups.items() if g};assert not set(groups).intersection(('3','4','11','13','15','16'))
    cv={};recipes=('source_normalized','nonoverlap_per_teacher')
    for recipe in recipes:
        cv[recipe]={};pooled=[]
        for n,held in groups.items():
            model=fitted([g for k,g in groups.items() if k!=n],recipe);e=errors(model,held);pooled.extend(e);cv[recipe][n]=dict(n=len(e),mean=float(np.mean(e)))
        cv[recipe]['pooled']=dict(n=len(pooled),mean=float(np.mean(pooled)),median=float(np.median(pooled)));print('TRAIN CV',recipe,cv[recipe]['pooled'],flush=True)
    chosen=min(recipes,key=lambda k:cv[k]['pooled']['mean']);model=fitted(list(groups.values()),chosen);model.update(training_manifest={n:len(g) for n,g in groups.items()},native_teachers_available=sum(map(len,groups.values())))
    (out/'training_selection.json').write_text(json.dumps(dict(cv=cv,chosen=chosen,evaluation_data_read_for_selection=False),indent=2),'utf8')
    (out/'shared_model.json').write_text(json.dumps(model,indent=2),'utf8');print('MODEL FROZEN',chosen,flush=True)
    files=['tools/evaluate_likelihood414.py','visibility_response.py','conditional_unknown_force.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    evaluation=('13','15','16','11','3','4')
    for n in evaluation:
        rr=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()];pred=VisibilityResponse(model);vv=[pred.step(r) for r in rr]
        assert all(r['valid']==old['valid'] and r['coefficient']==old['coefficient'] and r['force']==old['force'] for r,old in zip(vv,rr))
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8');print(n,'inference frozen',flush=True)
    cases={};pool={a:[] for a in ('control','visibility402','selected')}
    for n in evaluation:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),visibility402=score(ROOT.parent/f'output/visibility402_20261110/{n}_visibility.jsonl',ref,origins),selected=score(out/f'{n}.jsonl',ref,origins))
        for a,r in cases[n].items():pool[a].extend(q['error_px'] for q in r['records'] if q['valid'])
        print(n,{a:r['error']['mean'] for a,r in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),within35=sum(e<=35 for e in v),within30=sum(e<=30 for e in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,chosen_recipe=chosen,source_sha256=hashes,
        all6_evaluation_sources_excluded_from_fit=True,all_inference_before_reference_reads=True,recipe_chosen_without_evaluation_scores=True,
        independent_new_holdout_claimed=False,RAW_FPS_certificate=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
