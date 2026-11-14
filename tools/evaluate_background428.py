"""Causal nuisance-background correction; no change to rotor detector or state."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from background_unknown_response import BackgroundMotion,OnlineBackgroundResponse
from pure_visibility170 import window
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    out=ROOT.parent/'output/background428_20261110';out.mkdir(exist_ok=False);cv2.setNumThreads(2)
    decl=ROOT.parent/'output/checkpoints/background428_predeclared_20261110.json';(out/'protocol.json').write_bytes(decl.read_bytes())
    files=['background_unknown_response.py','physical_publication.py','tools/evaluate_background428.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    ns=('13','4','15','16','11','3');diag={}
    for n in ns:
        rr=[json.loads(s) for s in (ROOT.parent/f'output/bounded426_20261110/{n}_observed.jsonl').read_text('utf8').splitlines()]
        cache=ROOT.parent/('output/frozen324_20261109/cache' if n=='4' else 'output/frozen_tangent287_20261108/cache' if n=='3' else 'output/motion181_20261107/cache_recording15' if n=='15' else f'output/phase185_20261107/cache_recording{n}')
        manifest=json.loads((cache/'manifest.json').read_text('utf8'))['frames'];bg=BackgroundMotion();learn=OnlineBackgroundResponse();values=[];qualified=0
        for i,r in enumerate(rr):
            e=np.load(cache/manifest[i]['file']) if n in ('3','4') else window(cache,r['t'])
            count=np.bincount(e['y'].astype(np.int64)*1280+e['x'],minlength=1280*720).reshape(720,1280).astype(np.float32)
            info=bg.step(r,(count,np.zeros_like(count)));new=learn.step(r,info);qualified+=bool(info['qualified'])
            assert new['body']==r['body'] and new['valid']==r['valid'] and new['force']==r['force'] and new['coefficient']==r['coefficient'];values.append(new)
            if i%60==0:print(n,i,'background-qualified',qualified,'matured',learn.matured,'beta',float(np.clip(learn.xy/(1+learn.xx),0,1)),flush=True)
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in values),'utf8')
        diag[n]=dict(background_qualified=qualified,matured=learn.matured,final_beta=float(np.clip(learn.xy/(1+learn.xx),0,1)));(out/'progress.json').write_text(json.dumps(diag,indent=2),'utf8');print(n,'inference frozen',flush=True)
    cases={};pool={a:[] for a in ('control410','bounded426','background428')}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control410=score(ROOT.parent/f'output/combined410_20261110/{n}_joint.jsonl',ref,origins),bounded426=score(ROOT.parent/f'output/bounded426_20261110/{n}_observed.jsonl',ref,origins),background428=score(out/f'{n}.jsonl',ref,origins))
        original={q['t'] for q in cases[n]['control410']['records'] if q['valid']}
        for a,sc in cases[n].items():
            v=[q['error_px'] for q in sc['records'] if q['valid'] and q['t'] in original];assert len(v)==len(original),(n,a);pool[a].extend(v);sc['same_original_origins']=dict(n=len(v),mean=float(np.mean(v)))
        print(n,{a:sc['same_original_origins'] for a,sc in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled_same_original_origins=pooled,diagnostics=diag,source_sha256=hashes,current_target_detector_and_state_unchanged=True,online_parameters_use_only_matured_native_observations=True,manual_labels_not_used_for_parameter_updates=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
