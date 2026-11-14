"""One observation-function correction, all fixed six sources and original102."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from native_fov_observation import VisibleEnvelopeResponse,project
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    out=ROOT.parent/'output/visible421_20261110';out.mkdir(exist_ok=False)
    assert (ROOT.parent/'output/observation420_20261110/results.json').exists()
    declaration=ROOT.parent/'output/checkpoints/visible_projection421_predeclared_20261110.json';(out/'protocol.json').write_bytes(declaration.read_bytes())
    files=['native_fov_observation.py','physical_publication.py','tools/evaluate_visible421.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    ns=('13','15','16','11','3','4');diag={}
    for n in ns:
        diag[n]={}
        for arm,source in [('projected410',ROOT.parent/f'output/combined410_20261110/{n}_joint.jsonl'),('projected420',ROOT.parent/f'output/observation420_20261110/{n}_observed.jsonl')]:
            rr=[json.loads(s) for s in source.read_text('utf8').splitlines()];pred=VisibleEnvelopeResponse();vv=[pred.step(r) for r in rr]
            assert all(r['valid']==old['valid'] and r['coefficient']==old['coefficient'] and r['force']==old['force'] for r,old in zip(vv,rr))
            changes=0;unclipped_error=0.
            for r,old in zip(vv,rr):
                if not r.get('native_fov_observation'):continue
                before=evaluate(old,[.4])['0.4'];after=evaluate(r,[.4])['0.4'];_,info=project(before,r['native_fov_observation']['full_extent_px'])
                if info.get('geometric_extent_clipped'):changes+=1
                else:unclipped_error=max(unclipped_error,float(np.max(abs(np.asarray(before)-after))))
            assert unclipped_error<1e-9;diag[n][arm]=dict(projected_clipped_frames=changes,interior_prediction_max_change_px=unclipped_error)
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(n,'inference frozen',flush=True)
    pool={a:[] for a in ('control','observed420','projected410','projected420')};cases={}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/combined410_20261110/{n}_joint.jsonl',ref,origins),observed420=score(ROOT.parent/f'output/observation420_20261110/{n}_observed.jsonl',ref,origins),**{a:score(out/f'{n}_{a}.jsonl',ref,origins) for a in ('projected410','projected420')})
        original={q['t'] for q in cases[n]['control']['records'] if q['valid']}
        for a,r in cases[n].items():
            v=[q['error_px'] for q in r['records'] if q['valid'] and q['t'] in original];pool[a].extend(v);r['same_original_origins']=dict(n=len(v),mean=float(np.mean(v)),p95=float(np.percentile(v,95)))
        print(n,{a:r['same_original_origins'] for a,r in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled_same_original_origins=pooled,diagnostics=diag,source_sha256=hashes,
        future_geometry_read=False,force_state_unchanged=True,original_references_unchanged=True,one_rule_all_sources=True,new_video_validation=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
