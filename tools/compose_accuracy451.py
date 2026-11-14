"""Freeze one diffuse-startup/window composition before reading references."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from window_position_observation import WindowPositionObservation
from score_rotor164 import score
from extend_inverse321 import references

if __name__=='__main__':
    out=ROOT.parent/'output/accuracy451_20261110';out.mkdir(exist_ok=False)
    files=['diffuse_rotor_startup.py','observed_rotor_predictor.py','run_rotor_accuracy_candidate.py','window_position_observation.py','physical_publication.py','tools/compose_accuracy451.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    (out/'protocol.json').write_text(json.dumps(dict(one_global_composition=True,no_source_switch=True,no_model_fit=True,
        original102_preserved=True,source_sha256=hashes,source_support_us=33333,target_support_us=20000),indent=2),'utf8')
    ns=('13','15','16','11','3','4')
    for n in ns:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/diffuse_startup450_20261110/{n}_diffuse.jsonl').read_text('utf8').splitlines()]
        window=WindowPositionObservation();vv=[window.step(r) for r in rows]
        assert all(all(r[k]==s[k] for k in ('valid','coefficient','force','body','t')) for r,s in zip(vv,rows))
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
    print('ALL INFERENCE FROZEN',flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ['complete447','accuracy451']}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a,p in [('complete447',ROOT.parent/f'output/reference_bootstrap447_20261110/temporal_{n}.jsonl'),('accuracy451',out/f'{n}.jsonl')]:
            s=score(p,ref,np.round(np.arange(10.8,15.6001,.2),1));v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids);pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,source_sha256=hashes,
        no_fit_or_search=True,original102_preserved=True,new_video_validated=False,FPS_work=False,weak_native_outline_not_COM=True),indent=2),'utf8');print('POOLED',pooled,flush=True)
