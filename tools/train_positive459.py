"""Align the same13 response parameters to the new prior, same186 teachers."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from replay_endpoint432 import run
from positive_normal_startup import PositiveStartupObserver
from plane_conditioned_response import plane_samples,fit_model,PlaneResponse
from future_drive_response import drive_feature
from reference_qualification import QualifiedReferenceResponse
from bootstrap_native_reference import BootstrapNativeReference
from native_fov_observation import VisibleEnvelopeResponse
from window_position_observation import WindowPositionObservation
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    out=ROOT.parent/'output/positive_training459_20261110';out.mkdir(exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(dict(no_parameter_search=True,teacher_origins_unchanged=True,
        all6_evaluation_recordings_excluded_from_fit=True,prior_mean_not_fitted=True),indent=2),'utf8')
    old=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));groups=[];manifest={}
    for n,count in old['training_manifest'].items():
        assert n not in ('13','15','16','11','3','4')
        rows=[json.loads(s) for s in (ROOT.parent/f'output/training400_20261110/recording{n}/predictions.jsonl').read_text('utf8').splitlines()]
        rr=run(rows,PositiveStartupObserver,old,calibrated=False);g=plane_samples(rr);expected=json.loads((ROOT.parent/f'output/plane409_20261110/teachers{n}.json').read_text('utf8'))
        assert [(s['t'],s['h']) for s in g]==[(s['t'],s['h']) for s in expected]
        lookup={r['t']:r for r in rr}
        for s in g:s['drive_feature']=drive_feature(lookup[s['t']],s['h']).tolist()
        groups.append(g);manifest[n]=len(g);(out/f'teachers{n}.json').write_text(json.dumps(g),'utf8');print(n,'training',len(g),flush=True)
    assert sum(manifest.values())==186
    model=fit_model(groups,collective_drive=True);model.update(training_manifest=manifest,all6_evaluation_sources_excluded_from_fit=True,positive_startup_required=True)
    (out/'model.json').write_text(json.dumps(model,indent=2),'utf8');print('MODEL FROZEN',flush=True)
    ns=('13','15','16','11','3','4')
    for n in ns:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/positive_normal457_20261110/{n}_positive.jsonl').read_text('utf8').splitlines()];response=QualifiedReferenceResponse(PlaneResponse(model));boot=BootstrapNativeReference();vis=VisibleEnvelopeResponse();win=WindowPositionObservation();vv=[]
        for oldrow in rows:
            r=copy.deepcopy(oldrow)
            for k in ('conditional_unknown_force','visibility_response','plane_conditioned_response','future_drive_response','native_fov_observation','window_position_observation','native_reference_bootstrap','physical'):r.pop(k,None)
            r=win.step(vis.step(boot.step(response.step(r))));assert r['coefficient']==oldrow['coefficient'] and r['force']==oldrow['force'] and r['valid']==oldrow['valid'];vv.append(r)
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
    print('ALL INFERENCE FROZEN',flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ('positive457','matched459')}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a,p in [('positive457',ROOT.parent/f'output/positive_normal457_20261110/{n}_positive.jsonl'),('matched459',out/f'{n}.jsonl')]:
            s=score(p,ref,np.round(np.arange(10.8,15.6001,.2),1));v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids);pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,training_manifest=manifest,model_sha256=hashlib.sha256((out/'model.json').read_bytes()).hexdigest(),all6_evaluation_sources_excluded_from_fit=True,new_video_validated=False,FPS_work=False,not_actual_RAW_factory_yet=True),indent=2),'utf8');print('POOLED',pooled,flush=True)
