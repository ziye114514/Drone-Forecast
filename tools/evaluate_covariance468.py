"""Matched-second-moment ablation: force mean vs input uncertainty benefit."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from equilibrium_gain_prior import CovarianceOnlyObserver
from replay_endpoint432 import run
from bootstrap_native_reference import BootstrapNativeReference
from native_fov_observation import VisibleEnvelopeResponse
from window_position_observation import WindowPositionObservation
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    out=ROOT.parent/'output/covariance468_20261111';out.mkdir(exist_ok=False)
    files=['equilibrium_gain_prior.py','tools/evaluate_covariance468.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,mean_kept_at_old_default=True,
        prior_gain_second_moment_matches463=True,no_fit_or_noise_search=True,all6_uniform=True,
        all_inference_before_references=True,experimental_ablation_not_measured_force=True),indent=2),'utf8')
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    model=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));ns=('13','15','16','11','3','4')
    for n in ns:
        for a,name in [('covariance','control'),('covariance_partial','partial')]:
            rows=[json.loads(s) for s in (ROOT.parent/f'output/partial_gain467_20261111/{n}_{name}.jsonl').read_text('utf8').splitlines()]
            rr=run(rows,CovarianceOnlyObserver,model);boot=BootstrapNativeReference();vis=VisibleEnvelopeResponse();win=WindowPositionObservation();vv=[win.step(vis.step(boot.step(r))) for r in rr]
            assert all(r['valid']==s['valid'] and r['body']==s['body'] and r['motion']==s['motion'] for r,s in zip(vv,rows))
            (out/f'{n}_{a}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(n,'inference frozen',flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ('gain463','partial467','covariance','covariance_partial')}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a,p in [('gain463',ROOT.parent/f'output/partial_gain467_20261111/{n}_control.jsonl'),('partial467',ROOT.parent/f'output/partial_gain467_20261111/{n}_partial.jsonl'),*[(a,out/f'{n}_{a}.jsonl') for a in ('covariance','covariance_partial')]]:
            s=score(p,ref,np.round(np.arange(10.8,15.6001,.2),1));v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids);pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within31=sum(x<=31 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,original102_preserved=True,new_video_validated=False,FPS_work=False,
        matched_second_moment_not_identical_conditional_posterior=True),indent=2),'utf8');print('POOLED',pooled,flush=True)
