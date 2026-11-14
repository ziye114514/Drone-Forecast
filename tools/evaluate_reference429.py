"""Qualification of output reference, not a new detector or force fit."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from reference_qualification import QualifiedReferenceResponse
from plane_conditioned_response import PlaneResponse
from native_fov_observation import VisibleEnvelopeResponse
from extend_inverse321 import references
from score_rotor164 import score
from physical_publication import evaluate


def base(row):
    r=copy.deepcopy(row)
    for k in ('conditional_unknown_force','visibility_response','plane_conditioned_response','future_drive_response','native_fov_observation','physical'):r.pop(k,None)
    return r


if __name__=='__main__':
    out=ROOT.parent/'output/reference429_20261110';out.mkdir(exist_ok=False);decl=ROOT.parent/'output/checkpoints/reference429_predeclared_20261110.json';(out/'protocol.json').write_bytes(decl.read_bytes())
    model=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));files=['reference_qualification.py','calibrated_force_rollout.py','conditional_unknown_force.py','native_fov_observation.py','tools/evaluate_reference429.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    ns=('13','4','15','16','11','3')
    for n in ns:
        rr=[json.loads(s) for s in (ROOT.parent/f'output/bounded426_20261110/{n}_observed.jsonl').read_text('utf8').splitlines()];cal=QualifiedReferenceResponse(PlaneResponse(model));visible=VisibleEnvelopeResponse();vv=[]
        for old in rr:
            q=visible.step(cal.step(base(old)));assert q['coefficient']==old['coefficient'] and q['force']==old['force'] and q['valid']==old['valid']
            assert all(q['body'].get(k)==old['body'].get(k) for k in ('detected','bbox','center'));vv.append(q)
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8');print(n,'inference frozen',flush=True)
    cases={};pool={a:[] for a in ('control410','bounded426','qualified429')}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control410=score(ROOT.parent/f'output/combined410_20261110/{n}_joint.jsonl',ref,origins),bounded426=score(ROOT.parent/f'output/bounded426_20261110/{n}_observed.jsonl',ref,origins),qualified429=score(out/f'{n}.jsonl',ref,origins))
        orig={q['t'] for q in cases[n]['control410']['records'] if q['valid']}
        for a,r in cases[n].items():
            v=[q['error_px'] for q in r['records'] if q['valid'] and q['t'] in orig];assert len(v)==len(orig);pool[a].extend(v);r['same_original_origins']=dict(n=len(v),mean=float(np.mean(v)))
        print(n,{a:r['same_original_origins'] for a,r in cases[n].items()},flush=True)
    # Independent semantic check: support changes without material motion.
    rr=[json.loads(s) for s in (ROOT.parent/'output/bounded426_20261110/15_observed.jsonl').read_text('utf8').splitlines()]
    seed=next(base(r) for r in rr if r['valid'] and r['body'].get('bbox') and min(r['body']['bbox'][:2])>100 and r['body']['bbox'][2]<1180 and r['body']['bbox'][3]<620)
    seed['coefficient'].update(velocity=[0.,0.],acceleration=[0.,0.]);seed['force']['acceleration_px_s2']=[0.,0.]
    for k in ('response_model_mixture','external_unknown_rate_px_s3'):seed['force'].pop(k,None)
    seed['normal_turn_forecast']=dict(applied=False);guard=QualifiedReferenceResponse(PlaneResponse(model));plain=PlaneResponse(model);v1=[];v2=[]
    for i in range(30):
        r=copy.deepcopy(seed);r['t']=100.+i/30;r['state']['observations']=i+200;r['periodic_track_continuation']=dict(used=i>0)
        delta=np.array([9*np.sin(i/4),6*np.cos(i/5)]) if i else np.zeros(2);r['body']['bbox']=(np.asarray(seed['body']['bbox'])+np.tile(delta,2)).tolist()
        v1.append(evaluate(guard.step(r),[.4])['0.4']);v2.append(evaluate(plain.step(r),[.4])['0.4'])
    invariant=float(np.max(abs(np.asarray(v1)-v1[0])));unqualified=float(np.max(abs(np.asarray(v2)-v2[0])));assert invariant<1e-8 and unqualified>1.
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled_same_original_origins=pooled,source_sha256=hashes,
        support_wobble_stationary_material_reference_control=dict(qualified_readout_max_shift_px=invariant,unguarded_readout_shift_px=unqualified),
        current_detector_and_state_unchanged=True,no_model_fit=True,FPS_work=False,new_video_validated=False),indent=2),'utf8');print('POOLED',pooled,'WOBBLE',invariant,unqualified,flush=True)
