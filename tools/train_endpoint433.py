"""Response model aligned to the shared-image observer; same186 teachers."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from replay_endpoint432 import run
from shared_endpoint_observer import EndpointAdaptiveObserver
from plane_conditioned_response import plane_samples,fit_model,PlaneResponse
from future_drive_response import drive_feature
from reference_qualification import QualifiedReferenceResponse
from native_fov_observation import VisibleEnvelopeResponse
from extend_inverse321 import references
from score_rotor164 import score


def strip(old):
    r=copy.deepcopy(old)
    for k in ('conditional_unknown_force','visibility_response','plane_conditioned_response','future_drive_response','native_fov_observation','physical'):r.pop(k,None)
    return r


if __name__=='__main__':
    out=ROOT.parent/'output/endpoint_training433_20261110';out.mkdir(exist_ok=False)
    (out/'protocol.json').write_bytes((ROOT.parent/'output/checkpoints/endpoint_training433_predeclared_20261110.json').read_bytes())
    files=['shared_endpoint_observer.py','tools/replay_endpoint432.py','tools/train_endpoint433.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    old=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));groups=[];manifest={}
    for n in old['training_manifest']:
        assert n not in ('13','15','16','11','3','4')
        rows=[json.loads(s) for s in (ROOT.parent/f'output/training400_20261110/recording{n}/predictions.jsonl').read_text('utf8').splitlines()]
        rr=run(rows,EndpointAdaptiveObserver,old,calibrated=False);expected=json.loads((ROOT.parent/f'output/plane409_20261110/teachers{n}.json').read_text('utf8'))
        g=plane_samples(rr);lookup={r['t']:r for r in rr}
        assert [(s['t'],s['h']) for s in g]==[(s['t'],s['h']) for s in expected],(n,'teacher origin changed')
        for s in g:s['drive_feature']=drive_feature(lookup[s['t']],s['h']).tolist()
        groups.append(g);manifest[n]=len(g)
        (out/f'teachers{n}.json').write_text(json.dumps(g),'utf8')
        (out/'progress.json').write_text(json.dumps(dict(training_manifest=manifest,stage='same186_teacher_replay',manual_references_read=False),indent=2),'utf8')
        print(n,'frozen native teacher rows',len(g),'packets',len(rr),flush=True)
    assert sum(manifest.values())==186
    model=fit_model(groups,collective_drive=True);model.update(training_manifest=manifest,all6_evaluation_sources_excluded_from_fit=True,
        shared_image_endpoint_observer_required=True,one_pair_displacement_variance_floor_px2=4.,endpoint_image_variance_px2=2.)
    (out/'shared_model.json').write_text(json.dumps(model,indent=2),'utf8');print('MODEL FROZEN',model['weights'],model['future_drive_multiplier'],flush=True)
    ns=('13','15','16','11','3','4')
    for n in ns:
        rr=[json.loads(s) for s in (ROOT.parent/f'output/endpoint432_20261110/{n}_endpoint.jsonl').read_text('utf8').splitlines()];response=QualifiedReferenceResponse(PlaneResponse(model));visible=VisibleEnvelopeResponse();vv=[]
        for oldrow in rr:
            r=visible.step(response.step(strip(oldrow)));assert r['coefficient']==oldrow['coefficient'] and r['force']==oldrow['force'] and r['valid']==oldrow['valid'];vv.append(r)
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8');print(n,'inference frozen',flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ('qualified429','fixed432','matched433')}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a,p in [('qualified429',ROOT.parent/f'output/reference429_20261110/{n}.jsonl'),('fixed432',ROOT.parent/f'output/endpoint432_20261110/{n}_endpoint.jsonl'),('matched433',out/f'{n}.jsonl')]:
            r=score(p,ref,origins);v=[q['error_px'] for q in r['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids);pool[a].extend(v);r['same_original_origins']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=r
        print(n,{a:r['same_original_origins'] for a,r in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled_same_original_origins=pooled,training_manifest=manifest,source_sha256=hashes,
        all6_evaluation_recordings_excluded_from_fit=True,teacher_origins_unchanged=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
