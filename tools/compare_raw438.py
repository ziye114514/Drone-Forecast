"""Fair old/new observer comparison on IDENTICAL actualRAW437 observations."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from replay_endpoint432 import run
from adaptive_response_observer import AdaptiveResponseObserver
from shared_endpoint_observer import ProjectionQualifiedObserver
from physical_publication import evaluate
from score_rotor164 import score
from extend_inverse321 import references

if __name__=='__main__':
    out=ROOT.parent/'output/raw_comparison438_20261110';out.mkdir(exist_ok=False)
    model=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));ns=('13','15','16','11','3','4');parity={}
    for n in ns:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/raw_projection437_20261110/recording{n}/predictions.jsonl').read_text('utf8').splitlines()]
        for a,cls in [('old429',AdaptiveResponseObserver),('new436',ProjectionQualifiedObserver)]:
            vv=run(rows,cls,model);assert all(r['valid']==s['valid'] and r['body']==s['body'] for r,s in zip(vv,rows))
            if a=='new436':
                d=max([float(np.max(abs(np.asarray(evaluate(r,[.4])['0.4'])-evaluate(s,[.4])['0.4']))) for r,s in zip(vv,rows) if r['valid']] or [0.]);assert d<1e-7,(n,d);parity[n]=d
            (out/f'{n}_{a}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(n,'sameRAW observations frozen, actual parity',parity[n],flush=True)
    cases={};pool={a:[] for a in ('old429','new436')};original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'))
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a in pool:
            r=score(out/f'{n}_{a}.jsonl',ref,origins);v=[q['error_px'] for q in r['records'] if q['valid'] and q['t'] in ids];pool[a].extend(v);r['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=r
        print(n,{a:r['original_available'] for a,r in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    assert pooled['old429']['n']==pooled['new436']['n']==101
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled_identical_RAW_original101=pooled,actual_entry_parity=parity,
        same_front_packets_and_startup=True,one_original102_missing=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
