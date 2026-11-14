"""One initial-prior correction; same six frozen fronts and original102."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from diffuse_rotor_startup import DiffuseStartupObserver
from shared_endpoint_observer import ProjectionQualifiedObserver
from replay_endpoint432 import run
from bootstrap_native_reference import BootstrapNativeReference
from native_fov_observation import VisibleEnvelopeResponse
from score_rotor164 import score
from extend_inverse321 import references
from physical_publication import evaluate


def controls():
    # A real second image, never a transported/birth-only search velocity.
    observers=[ProjectionQualifiedObserver(),DiffuseStartupObserver()]
    snapshots=[]
    for i in range(6):
        body=dict(detected=True,center=[200.+20*i,300.],measurement_source='filtered_native_translation' if i else 'rotor_region_initialization')
        motion=dict(current_observation=True,dt_s=1/30,estimate=dict(dx=20.,dy=0.,score=1.,current_displacement_qualified=True))
        for m in observers:m.step(1+i/30,body,None,input_meta=dict(observed_components={},input_sigma_proxy=0.),motion=motion)
        snapshots.append([dict(v=m.x[2:4].tolist(),a=m.x[4:6].tolist()) for m in observers])
        assert all(np.isfinite(m.P).all() and np.linalg.eigvalsh(m.P).min()>-1e-3 for m in observers)
    assert np.max(abs(observers[1].x[2:4]-[600.,0.]))<1.
    assert abs(snapshots[1][1]['a'][0])<.01
    assert abs(snapshots[1][0]['a'][0])>100.
    return snapshots


if __name__=='__main__':
    out=ROOT.parent/'output/diffuse_startup450_20261110';out.mkdir(exist_ok=False)
    files=['diffuse_rotor_startup.py','tools/evaluate_diffuse450.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    (out/'protocol.json').write_text(json.dumps(dict(initial_velocity_sigma_px_s=1e6,prior_selected_without_score_search=True,
        no_model_fit=True,front_unchanged=True,reference_unchanged=True,scored_after_all_inference=True,source_sha256=hashes),indent=2),'utf8')
    toy=controls();model=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));ns=('13','15','16','11','3','4');parity={}
    for n in ns:
        path=ROOT.parent/f'output/temporal444_20261110/{n}_temporal.jsonl';rr=[json.loads(s) for s in path.read_text('utf8').splitlines()]
        for arm,cls in [('control',ProjectionQualifiedObserver),('diffuse',DiffuseStartupObserver)]:
            vv=run(rr,cls,model);boot=BootstrapNativeReference();vis=VisibleEnvelopeResponse();vv=[vis.step(boot.step(r)) for r in vv]
            assert all(r['valid']==s['valid'] and r['body']==s['body'] and r['motion']==s['motion'] for r,s in zip(vv,rr))
            if arm=='control':
                expected=[json.loads(s) for s in (ROOT.parent/f'output/reference_bootstrap447_20261110/temporal_{n}.jsonl').read_text('utf8').splitlines()]
                delta=max([np.max(abs(np.asarray(evaluate(r,[.4])['0.4'])-evaluate(s,[.4])['0.4'])) for r,s in zip(vv,expected) if r['valid']] or [0.]);assert delta<1e-7,(n,delta);parity[n]=float(delta)
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(n,'inference frozen',flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ['control','diffuse']}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for arm in pool:
            s=score(out/f'{n}_{arm}.jsonl',ref,np.round(np.arange(10.8,15.6001,.2),1));v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids);pool[arm].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][arm]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,control_parity=parity,synthetic=toy,source_sha256=hashes,original102_preserved=True,no_fit_or_prior_search=True,FPS_work=False,new_video_validated=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
