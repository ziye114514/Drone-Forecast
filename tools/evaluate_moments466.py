"""Closed-form prior moments, same existing mean, no prior-variance search."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from positive_normal_moments import MomentObserver,MomentGainObserver
from positive_normal_startup import PositiveStartupObserver
from equilibrium_gain_prior import EquilibriumGainObserver,anchor_metadata
from replay_endpoint432 import run
from bootstrap_native_reference import BootstrapNativeReference
from native_fov_observation import VisibleEnvelopeResponse
from window_position_observation import WindowPositionObservation
from extend_inverse321 import references
from score_rotor164 import score
from physical_publication import evaluate

if __name__=='__main__':
    out=ROOT.parent/'output/positive_moments466_20261111';out.mkdir(exist_ok=False)
    files=['positive_normal_moments.py','positive_normal_startup.py','normal_response_observer.py','tools/evaluate_moments466.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,half_normal_exact_startup_moments=True,
        untruncated_sigma_unchanged=1000.,mean_unchanged=1000*np.sqrt(2/np.pi),variance_not_fitted=1e6*(1-2/np.pi),
        all6_uniform=True,no_fit_or_variance_search=True,all_inference_before_references=True),indent=2),'utf8')
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    a=PositiveStartupObserver();b=MomentObserver();a._initialize(np.array([200.,300.]));b._initialize(np.array([200.,300.]));assert np.array_equal(a.x,b.x) and np.array_equal(a.P[:7,:7],b.P[:7,:7]);assert abs(b.P[b.response_index,b.response_index]-1e6*(1-2/np.pi))<1e-7
    model=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));ns=('13','15','16','11','3','4');parity={}
    arms={'old_positive':PositiveStartupObserver,'old_gain':EquilibriumGainObserver,'moments':MomentObserver,'gain_moments':MomentGainObserver}
    for n in ns:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/positive_normal457_20261110/{n}_positive.jsonl').read_text('utf8').splitlines()];modified=[]
        for old in rows:
            r=copy.deepcopy(old);r['period_selection']=anchor_metadata(r['period_selection'],r.get('period',{}).get('selected_frequency_hz'));modified.append(r)
        parity[n]={}
        for arm,cls in arms.items():
            rr=run(modified,cls,model);boot=BootstrapNativeReference();vis=VisibleEnvelopeResponse();win=WindowPositionObservation();vv=[win.step(vis.step(boot.step(r))) for r in rr]
            assert all(r['valid']==s['valid'] and r['body']==s['body'] and r['motion']==s['motion'] for r,s in zip(vv,rows))
            if arm in ('old_positive','old_gain'):
                expected=rows if arm=='old_positive' else [json.loads(s) for s in (ROOT.parent/f'output/equilibrium463_20261111/{n}_equilibrium.jsonl').read_text('utf8').splitlines()]
                delta=max([np.max(abs(np.array(evaluate(r,[.4])['0.4'])-evaluate(s,[.4])['0.4'])) for r,s in zip(vv,expected) if r['valid']] or [0.]);assert delta<1e-7,(n,arm,delta);parity[n][arm]=float(delta)
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(n,'all inference frozen',flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in arms}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a in pool:
            s=score(out/f'{n}_{a}.jsonl',ref,np.round(np.arange(10.8,15.6001,.2),1));v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids);pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within31=sum(x<=31 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,control_parity=parity,original102_preserved=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
