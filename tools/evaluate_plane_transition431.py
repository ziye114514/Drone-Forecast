"""One two-column trial; inference frozen before unchanged102 scoring."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from plane_transition_response import basis,apply
from plane_conditioned_response import PlaneResponse
from conditional_unknown_force import fit,HORIZON,TAU
from separated_force_response import rate_kernel
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score


if __name__=='__main__':
    out=ROOT.parent/'output/plane_transition431_20261110';out.mkdir(exist_ok=False)
    (out/'protocol.json').write_bytes((ROOT.parent/'output/checkpoints/plane_transition431_predeclared_20261110.json').read_bytes())
    files=['plane_transition_response.py','tools/evaluate_plane_transition431.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    old=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));groups=[];manifest={}
    for n in old['training_manifest']:
        assert n not in ('13','4','15','16','11','3')
        rows=[json.loads(s) for s in (ROOT.parent/f'output/training400_20261110/recording{n}/predictions.jsonl').read_text('utf8').splitlines()]
        response=PlaneResponse(old);processed={r['t']:response.step(r) for r in rows}
        teachers=json.loads((ROOT.parent/f'output/plane409_20261110/teachers{n}.json').read_text('utf8'));group=[]
        for s in teachers:
            r=processed[s['t']];h=s['h'];factor=rate_kernel(-1/TAU,0.,h)/rate_kernel(-1/TAU,0.,HORIZON)
            box=np.asarray(r['body']['bbox'],float);origin=(box[:2]+box[2:])/2
            baseline=np.asarray(evaluate(r,[h])[str(float(h))])-origin
            original=next(q for q in rows if q['t']==s['t'])
            # Saved teacher y is a residual. Reconstruct native displacement
            # from the SAME original physical rollout used in409.
            target=np.asarray(s['y'])+np.asarray(evaluate(original,[h])[str(float(h))])-np.asarray(original['coefficient']['position'])
            group.append(dict(t=s['t'],h=h,X=(basis(r)*factor).tolist(),y=(target-baseline).tolist()))
        groups.append(group);manifest[n]=len(group)
    assert sum(manifest.values())==186
    model=fit(groups,'camera',prior=np.zeros(2));weights=model['weights'];(out/'model.json').write_text(json.dumps(dict(model,training_manifest=manifest,frozen_base_model=old),indent=2),'utf8');print('FROZEN',weights,flush=True)
    ns=('13','15','16','11','3','4')
    for n in ns:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/reference429_20261110/{n}.jsonl').read_text('utf8').splitlines()]
        vv=[apply(r,weights) for r in rows]
        assert all(r['coefficient']==s['coefficient'] and r['force']==s['force'] and r['valid']==s['valid'] for r,s in zip(vv,rows))
        for r in rows:
            q=copy.deepcopy(r)
            if q['structure'].get('axis_xy'):q['structure']['axis_xy']=(-np.asarray(q['structure']['axis_xy'])).tolist()
            assert np.allclose(basis(q),basis(r),atol=1e-10)
            if q['coefficient']:
                q['coefficient']['velocity']=[0.,0.];q['coefficient']['acceleration']=[0.,0.];assert np.max(abs(basis(q)))==0.
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8');print(n,'inference frozen',flush=True)
    cases={};pool={k:[] for k in ('qualified429','transition431')}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'))['cases'][n]['control410']['records'];ids={q['t'] for q in original if q['valid']}
        cases[n]={}
        for k,p in [('qualified429',ROOT.parent/f'output/reference429_20261110/{n}.jsonl'),('transition431',out/f'{n}.jsonl')]:
            r=score(p,ref,origins);v=[q['error_px'] for q in r['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids);pool[k].extend(v)
            r['same_original_origins']=dict(n=len(v),mean=float(np.mean(v)));cases[n][k]=r
        print(n,{k:v['same_original_origins'] for k,v in cases[n].items()},flush=True)
    pooled={k:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for k,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled_same_original_origins=pooled,source_sha256=hashes,training_manifest=manifest,
        no_manual_reference_fit=True,current_state_force_validity_unchanged=True,FPS_work=False,new_video_validated=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
