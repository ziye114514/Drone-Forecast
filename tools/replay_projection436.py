"""Fix an unobserved inclination switch without deleting current azimuth."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from normal_response_observer import NormalResponseObserver
from shared_endpoint_observer import EndpointAdaptiveObserver,ProjectionQualifiedObserver
from replay_endpoint432 import run
from extend_inverse321 import references
from score_rotor164 import score
from physical_publication import evaluate


class QualifiedNormal(NormalResponseObserver):preserve_unmeasured_projection=True


def control(cls):
    m=cls();meta=dict(observed_components={},input_sigma_proxy=0.)
    structure=dict(mode='partial_two',pose_source='current_qualified_visible_conics',axis_xy=[0.,-.6])
    for i in range(8):
        t=1+i/30;b=dict(detected=True,center=[500.,300.],center_provisional=False,measurement_source='filtered_native_translation')
        motion=dict(dt_s=1/30,current_observation=True,estimate=dict(dx=0.,dy=0.,score=1.))
        m.step(t,b,None,input_meta=meta,motion=motion,structure=structure)
    m.x[:6]=[500.,300.,0.,0.,0.,0.];m.x[7:9]=[500.,300.];m.x[m.response_index]=1000.;m.normal=np.array([0.,-.6]);m.normal_observation_kind='current_conic_projection'
    # No correlation of hidden amplitude with the new position innovation.
    m.P[m.response_index]=0.;m.P[:,m.response_index]=0.;m.P[m.response_index,m.response_index]=1e6
    side=dict(mode='axis_only',axis_xy=[0.,-1.],axis_support=dict(valid=True,confidence=.9))
    c,_,f,s=m.step(m.t+1/30,b,None,input_meta=meta,motion=motion,structure=side)
    first=np.asarray(c['acceleration']);amp=m.x[m.response_index]
    c,_,f,s=m.step(m.t+1/30,b,None,input_meta=meta,motion=motion,structure=structure)
    roundtrip=c['acceleration']
    turned=dict(structure,axis_xy=[.6*np.sin(.05),-.6*np.cos(.05)])
    c,_,f,s=m.step(m.t+1/30,b,None,input_meta=meta,motion=motion,structure=turned)
    return dict(first_acceleration=first.tolist(),first_projected_amplitude=float(amp),roundtrip_acceleration=roundtrip,
        real_azimuth_change_acceleration=c['acceleration'])


if __name__=='__main__':
    out=ROOT.parent/'output/projection436_fixed_20261110';out.mkdir(exist_ok=False)
    (out/'protocol.json').write_bytes((ROOT.parent/'output/checkpoints/projection_gauge436_predeclared_20261110.json').read_bytes())
    checks=dict(old=control(NormalResponseObserver),qualified=control(QualifiedNormal));assert np.linalg.norm(checks['old']['first_acceleration'])>399
    assert np.linalg.norm(checks['qualified']['first_acceleration'])<1e-8 and np.linalg.norm(checks['qualified']['roundtrip_acceleration'])<1e-8;assert abs(checks['qualified']['first_projected_amplitude']-600)<1e-7
    assert np.linalg.norm(checks['qualified']['real_azimuth_change_acceleration'])>29.
    files=['normal_response_observer.py','shared_endpoint_observer.py','tools/replay_projection436.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    (out/'controls.json').write_text(json.dumps(checks,indent=2),'utf8');print('CONTROLS',checks,flush=True)
    model=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));ns=('13','15','16','11','3','4');parity={}
    for n in ns:
        # Original429 preserves the RAW factory's partial-entry startup flag.
        # The first432 serialized output omitted that bookkeeping field; its
        # own inference/RAW parity was correct but it is not replay-complete.
        rows=[json.loads(s) for s in (ROOT.parent/f'output/reference429_20261110/{n}.jsonl').read_text('utf8').splitlines()]
        wanted=[json.loads(s) for s in (ROOT.parent/f'output/endpoint432_20261110/{n}_endpoint.jsonl').read_text('utf8').splitlines()]
        for arm,cls in [('control',EndpointAdaptiveObserver),('qualified',ProjectionQualifiedObserver)]:
            vv=run(rows,cls,model);assert all(r['body']==s['body'] and r['valid']==s['valid'] and r['motion']==s['motion'] for r,s in zip(vv,rows))
            if arm=='control':
                d=max([float(np.max(abs(np.asarray(evaluate(r,[.4])['0.4'])-evaluate(s,[.4])['0.4']))) for r,s in zip(vv,wanted) if r['valid']] or [0.]);assert d<1e-7,(n,d);parity[n]=d
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(n,'inference frozen, default control delta',parity[n],flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ('control','qualified')}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a in pool:
            r=score(out/f'{n}_{a}.jsonl',ref,origins);v=[q['error_px'] for q in r['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids);pool[a].extend(v);r['same_original_origins']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=r
        print(n,{a:r['same_original_origins'] for a,r in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled_same_original_origins=pooled,controls=checks,control_parity=parity,source_sha256=hashes,
        one_global_rule=True,same_observations=True,no_model_fit=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
