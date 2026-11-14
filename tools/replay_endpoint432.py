"""Frozen front, two replayed observers, unchanged shared model and102 origins."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from shared_endpoint_observer import EndpointAdaptiveObserver
from adaptive_response_observer import AdaptiveResponseObserver
from inverse_normal_response import InverseNormalResponse
from anisotropic_response import AnisotropicResponse
from plane_conditioned_response import PlaneResponse
from reference_qualification import QualifiedReferenceResponse
from native_fov_observation import VisibleEnvelopeResponse
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score


def run(rows,cls,model,*,calibrated=True):
    observer=cls();inv=InverseNormalResponse();disk=AnisotropicResponse();response=QualifiedReferenceResponse(PlaneResponse(model));visible=VisibleEnvelopeResponse();outputs=[]
    for old in rows:
        r=copy.deepcopy(old)
        for k in ('conditional_unknown_force','visibility_response','plane_conditioned_response','future_drive_response','native_fov_observation','physical','window_position_observation','native_reference_bootstrap'):r.pop(k,None)
        u=old['force'].get('input_u');u=None if u is None else np.asarray(u,float);meta=r['period_selection']
        fresh=observer.x is None or observer.seen is not None and r['t']-observer.seen>.5
        if fresh and r['body'].get('detected') and not r['body'].get('center_provisional'):observer.two_point_init=bool(r['state'].get('two_point_initialization_selected_by_partial_entry'))
        c,v,f,s=observer.step(r['t'],r['body'],u,.7 if u is not None else 0.,allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,motion=r['motion'],structure=r['structure'])
        s['two_point_initialization_selected_by_partial_entry']=bool(old['state'].get('two_point_initialization_selected_by_partial_entry'))
        r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'))
        r=disk.step(inv.step(r))
        if calibrated:r=visible.step(response.step(r));r['physical']=evaluate(r,[.1,.2,.4]) if c else {}
        outputs.append(r)
    return outputs


if __name__=='__main__':
    out=ROOT.parent/'output/endpoint432_20261110';out.mkdir(exist_ok=False)
    (out/'protocol.json').write_bytes((ROOT.parent/'output/checkpoints/shared_endpoint432_predeclared_20261110.json').read_bytes())
    files=['shared_endpoint_observer.py','tools/replay_endpoint432.py','tools/check_endpoint432.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    model=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));ns=('13','15','16','11','3','4');parity={}
    for n in ns:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/reference429_20261110/{n}.jsonl').read_text('utf8').splitlines()]
        for arm,cls in [('control',AdaptiveResponseObserver),('endpoint',EndpointAdaptiveObserver)]:
            vv=run(rows,cls,model);assert all(r['valid']==s['valid'] and r['body']==s['body'] and r['motion']==s['motion'] for r,s in zip(vv,rows))
            if arm=='control':
                d=max([max(abs(np.asarray(evaluate(r,[.4])['0.4'])-evaluate(s,[.4])['0.4'])) for r,s in zip(vv,rows) if r['valid']] or [0.]);assert d<1e-7,(n,d);parity[n]=float(d)
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(n,'inference frozen, control delta',parity[n],flush=True)
    cases={};pool={a:[] for a in ('control','endpoint')};original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'))
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for arm in pool:
            r=score(out/f'{n}_{arm}.jsonl',ref,origins);v=[q['error_px'] for q in r['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids);pool[arm].extend(v);r['same_original_origins']=dict(n=len(v),mean=float(np.mean(v)));cases[n][arm]=r
        print(n,{a:r['same_original_origins'] for a,r in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled_same_original_origins=pooled,control_parity=parity,source_sha256=hashes,
        same_observations=True,one_pair_noise_floor_preserved=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
