"""Single declared normal-response prior; no fit or grid search."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from positive_normal_startup import PositiveStartupObserver
from diffuse_rotor_startup import DiffuseStartupObserver
from replay_endpoint432 import run
from bootstrap_native_reference import BootstrapNativeReference
from native_fov_observation import VisibleEnvelopeResponse
from window_position_observation import WindowPositionObservation
from extend_inverse321 import references
from score_rotor164 import score
from physical_publication import evaluate

if __name__=='__main__':
    out=ROOT.parent/'output/positive_normal457_20261110';out.mkdir(exist_ok=False)
    files=['positive_normal_startup.py','tools/evaluate_positive457.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,normal_prior_mean=1000*np.sqrt(2/np.pi),
        prior_variance_unchanged=1e6,current_net_acceleration_unchanged=True,not_measured_thrust=True,
        no_prior_search=True,no_fit=True,all_inference_before_references=True),indent=2),'utf8')
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    old=DiffuseStartupObserver();new=PositiveStartupObserver();old._initialize(np.array([200.,300.]));new._initialize(np.array([200.,300.]));assert np.array_equal(old.x[:7],new.x[:7]) and np.array_equal(old.P,new.P)
    assert new.x[new.response_index]>0 and np.array_equal(new.x[4:6],[0.,0.])
    model=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));ns=('13','15','16','11','3','4');parity={}
    for n in ns:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/accuracy451_20261110/{n}.jsonl').read_text('utf8').splitlines()]
        for arm,cls in [('control',DiffuseStartupObserver),('positive',PositiveStartupObserver)]:
            rr=run(rows,cls,model);boot=BootstrapNativeReference();vis=VisibleEnvelopeResponse();win=WindowPositionObservation();vv=[win.step(vis.step(boot.step(r))) for r in rr]
            assert all(r['valid']==s['valid'] and r['body']==s['body'] and r['motion']==s['motion'] for r,s in zip(vv,rows))
            if arm=='control':
                delta=max([np.max(abs(np.asarray(evaluate(r,[.4])['0.4'])-evaluate(s,[.4])['0.4'])) for r,s in zip(vv,rows) if r['valid']] or [0.]);assert delta<1e-7,(n,delta);parity[n]=float(delta)
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(n,'inference frozen',flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ['control','positive']}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a in pool:
            s=score(out/f'{n}_{a}.jsonl',ref,np.round(np.arange(10.8,15.6001,.2),1));v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids);pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,control_parity=parity,original102_preserved=True,
        not_actual_RAW_factory_yet=True,new_video_validated=False,FPS_work=False,source_sha256=hashes),indent=2),'utf8');print('POOLED',pooled,flush=True)
