"""One physical initialization hypothesis, unchanged102 scoring origins."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from equilibrium_gain_prior import EquilibriumGainObserver,anchor_metadata
from positive_normal_startup import PositiveStartupObserver
from replay_endpoint432 import run
from bootstrap_native_reference import BootstrapNativeReference
from native_fov_observation import VisibleEnvelopeResponse
from window_position_observation import WindowPositionObservation
from extend_inverse321 import references
from score_rotor164 import score
from physical_publication import evaluate

if __name__=='__main__':
    out=ROOT.parent/'output/equilibrium463_20261111';out.mkdir(exist_ok=False)
    files=['equilibrium_gain_prior.py','tools/evaluate_equilibrium462.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,one_prior_all6=True,old_normal_prior457_unchanged=True,
        no_fit_or_prior_search=True,hidden_frequency_measurements_created=False,all_inference_before_references=True),indent=2),'utf8')
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    # A parameter-prior change may not inject net acceleration or falsely
    # tighten its covariance before any new observation has been assimilated.
    toy=EquilibriumGainObserver();toy._initialize(np.array([200.,300.]));toy.normal=np.array([0.,-1.]);toy.u=np.array([.01,-.03])
    before=toy.x[4:6]+toy.x[toy.gain_index]*toy.u;G=np.zeros((2,toy.state_dim));G[:,4:6]=np.eye(2);G[:,toy.gain_index]=toy.u;cov=G@toy.P@G.T
    toy._gain_learning_allowed(True,False,0,0.,False,dict(normal_gain_prior_anchor=dict(frequency_squared_hz2=180.**2,reference_hz=300.)))
    assert np.max(abs(before-toy.x[4:6]-toy.x[toy.gain_index]*toy.u))<1e-8
    assert np.max(abs(G@toy.P@G.T-cov))<1e-7
    model=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));ns=('13','15','16','11','3','4');parity={};evidence={}
    for n in ns:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/positive_normal457_20261110/{n}_positive.jsonl').read_text('utf8').splitlines()];modified=[]
        for old in rows:
            r=copy.deepcopy(old);r['period_selection']=anchor_metadata(r['period_selection'],r.get('period',{}).get('selected_frequency_hz'));modified.append(r)
        for arm,cls in [('control',PositiveStartupObserver),('equilibrium',EquilibriumGainObserver)]:
            rr=run(modified,cls,model);boot=BootstrapNativeReference();vis=VisibleEnvelopeResponse();win=WindowPositionObservation();vv=[win.step(vis.step(boot.step(r))) for r in rr]
            assert all(r['valid']==s['valid'] and r['body']==s['body'] and r['motion']==s['motion'] for r,s in zip(vv,rows))
            if arm=='control':
                delta=max([np.max(abs(np.array(evaluate(r,[.4])['0.4'])-evaluate(s,[.4])['0.4'])) for r,s in zip(vv,rows) if r['valid']] or [0.]);assert delta<1e-7,(n,delta);parity[n]=float(delta)
            else:evidence[n]=dict(gain_prior_applied_frames=sum(bool(r['state'].get('equilibrium_gain_prior')) for r in vv),gain_updates=sum(r['force'].get('gain_learned',False) for r in vv))
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(n,'frozen',evidence[n],flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ('control','equilibrium')}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a in pool:
            s=score(out/f'{n}_{a}.jsonl',ref,np.round(np.arange(10.8,15.6001,.2),1));v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids);pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within31=sum(x<=31 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,control_parity=parity,evidence=evidence,original102_preserved=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
