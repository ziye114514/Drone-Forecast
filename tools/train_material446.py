"""Same186 candidate teacher slots; only observed material motion teaches force."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from material_motion_teachers import material_displacement
from replay_endpoint432 import run
from shared_endpoint_observer import ProjectionQualifiedObserver
from plane_conditioned_response import plane_samples,fit_model,PlaneResponse
from future_drive_response import drive_feature
from reference_qualification import QualifiedReferenceResponse
from native_fov_observation import VisibleEnvelopeResponse
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score


def strip(old):
    r=copy.deepcopy(old)
    for k in ('conditional_unknown_force','visibility_response','plane_conditioned_response','future_drive_response','native_fov_observation','physical'):r.pop(k,None)
    return r


if __name__=='__main__':
    out=ROOT.parent/'output/material_teachers446_20261110';out.mkdir(exist_ok=False)
    (out/'protocol.json').write_bytes((ROOT.parent/'output/checkpoints/material_teachers446_predeclared_20261110.json').read_bytes())
    files=['material_motion_teachers.py','tools/train_material446.py','tools/replay_endpoint432.py','shared_endpoint_observer.py','normal_response_observer.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    # Independent label semantics: changing support cannot change displacement.
    toy=[dict(t=1.,body=dict(center=[100.,200.]),motion=dict(current_observation=True)),dict(t=1.1,body=dict(center=[130.,195.],measurement_source='expanded_rotor_support_reference',measurement_reference_shift_xy=[20.,-5.],bbox=[1,2,3,4]),motion=dict(current_observation=True,dt_s=.1,estimate=dict(dx=10.,dy=0.)))]
    y,_=material_displacement(toy,0,1);assert np.max(abs(y-[10.,0.]))<1e-8
    toy[1]['body']['bbox']=[200,300,450,650];z,_=material_displacement(toy,0,1);assert np.array_equal(z,y)
    toy[1]['body']['measurement_source']='first_complete_rotor_region_anchor';assert material_displacement(toy,0,1)[0] is None
    old=json.loads((ROOT/'models/native_response_plane_v1.json').read_text('utf8'));groups={a:[] for a in ('box','material')};manifest={};qualified={};evidence={}
    for n,count in old['training_manifest'].items():
        assert n not in ('13','15','16','11','3','4');manifest[n]=count;qualified[n]=0
        if not count:continue
        rows=[json.loads(s) for s in (ROOT.parent/f'output/training400_20261110/recording{n}/predictions.jsonl').read_text('utf8').splitlines()]
        rr=run(rows,ProjectionQualifiedObserver,old,calibrated=False);g=plane_samples(rr);expected=json.loads((ROOT.parent/f'output/plane409_20261110/teachers{n}.json').read_text('utf8'))
        assert [(s['t'],s['h']) for s in g]==[(s['t'],s['h']) for s in expected]
        stamps=np.array([r['t'] for r in rr]);index={r['t']:i for i,r in enumerate(rr)};a=[];b=[];details=[]
        for sample in g:
            s=copy.deepcopy(sample);i=index[s['t']];j=int(np.argmin(abs(stamps-s['t']-s['h'])));r=rr[i];s['drive_feature']=drive_feature(r,s['h']).tolist();a.append(s)
            target,proof=material_displacement(rows,i,j);q=copy.deepcopy(s)
            if target is None:
                q.update(X=np.zeros((2,5)).tolist(),plane_X=np.zeros((2,2)).tolist(),drive_feature=[0.,0.],y=[0.,0.],label_information_available=False)
            else:
                baseline=np.asarray(evaluate(r,[s['h']])[str(float(s['h']))])-np.asarray(r['coefficient']['position'])
                q.update(y=(target-baseline).tolist(),label_information_available=True);qualified[n]+=1
            b.append(q);details.append(dict(t=s['t'],h=s['h'],proof=proof,material_displacement=None if target is None else target.tolist()))
        groups['box'].append(a);groups['material'].append(b);evidence[n]=details
        (out/f'teachers{n}_material.json').write_text(json.dumps(b),'utf8');print(n,'normalization slots',count,'real material labels',qualified[n],flush=True)
        (out/'progress.json').write_text(json.dumps(dict(normalization_slots=manifest,qualified_labels=qualified),indent=2),'utf8')
    assert sum(manifest.values())==186
    models={a:fit_model(g,collective_drive=True) for a,g in groups.items()}
    for a,m in models.items():
        m.update(training_manifest=manifest,all6_evaluation_sources_excluded_from_fit=True,projection_qualified_observer_required=True)
        if a=='material':m.update(training_samples=sum(qualified.values()),normalization_slots=186,qualified_labels_by_source=qualified,
            unqualified_slots_zero_information_not_zero_motion=True,material_motion_teacher=True,source_denominators_preserved=True)
        (out/f'{a}_model.json').write_text(json.dumps(m,indent=2),'utf8')
    (out/'teacher_evidence.json').write_text(json.dumps(evidence,indent=2),'utf8');print('MODELS FROZEN',[(a,m['training_samples'],m['future_drive_multiplier']) for a,m in models.items()],flush=True)
    ns=('13','15','16','11','3','4');paths={}
    for mode in ('cached','actual'):
        for n in ns:
            p=ROOT.parent/(f'output/projection436_fixed_20261110/{n}_qualified.jsonl' if mode=='cached' else f'output/raw_projection437_20261110/recording{n}/predictions.jsonl');paths[mode,n]=p
            rr=[json.loads(s) for s in p.read_text('utf8').splitlines()]
            for a,m in models.items():
                response=QualifiedReferenceResponse(PlaneResponse(m));visible=VisibleEnvelopeResponse();vv=[]
                for oldrow in rr:
                    r=visible.step(response.step(strip(oldrow)));assert r['coefficient']==oldrow['coefficient'] and r['force']==oldrow['force'] and r['valid']==oldrow['valid'];vv.append(r)
                (out/f'{mode}_{n}_{a}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in vv),'utf8')
        print(mode,'all inference frozen',flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pooled={}
    for mode in ('cached','actual'):
        pool={a:[] for a in ('control','box','material')};cases[mode]={}
        for n in ns:
            ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[mode][n]={}
            for a,p in [('control',paths[mode,n]),*[(a,out/f'{mode}_{n}_{a}.jsonl') for a in models]]:
                s=score(p,ref,origins);v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[mode][n][a]=s
            print(mode,n,{a:s['original_available'] for a,s in cases[mode][n].items()},flush=True)
        pooled[mode]={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within30=sum(x<=30 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,source_sha256=hashes,original_slots=186,real_material_labels=sum(qualified.values()),
        all6_recordings_excluded_from_training=True,no_hyperparameter_search=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
