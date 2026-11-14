"""Preregistered missing-observation pilot, not physical pixel occlusion."""
from pathlib import Path
import argparse,collections,copy,datetime,hashlib,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage119'),str(ROOT/'research/stage113'),str(ROOT/'research/stage110')]
from partial119 import PartialModel,PARAMS,geom,project
from study113 import stable,CenterBank,NAMES
from audit109 import rows,save,sha
OLD=ROOT.parent/'output/stage110_merge_safe_geometry_20261022'
REFERENCE=ROOT.parent/'output/stage117_persistent_structure_20261022'
OUT=ROOT.parent/'output/stage119_partial_conics_20261023'

def sources():
    paths=[Path(__file__),ROOT/'research/stage119/partial119.py',ROOT/'research/stage117/fixed117.py',ROOT/'research/stage113/support113.py',ROOT/'research/stage104/rigid104.py',ROOT/'research/stage107/center107.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def freeze():
    assert not OUT.exists()
    save(OUT/'protocol119.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),parameters=PARAMS,
        input_sha256={c:sha(OLD/c/'records.jsonl') for c in ['clear6','side6','negative5']},
        design='Four independent clear masks, remove the angle-sorted rotor rank0/1/2/3 only when exactly four current candidates exist. Intervention indices65..124 and155..184; first200 ticks. No fourth ellipse/fit is supplied to any masked estimator. Additional natural side100/negative30 ticks. Prefix100 rerun per mask.',
        controls='Known center projective-circle sequences: stationary wobble, moving wobble, abrupt turn; each150 frames with single missing observation45..99. Near-edge-on35..104 and two missing45..99. No stationary-motion prior. No model initialization from three; prefix causality and input permutation invariance.',
        evaluation='Outputs generated before evaluation. Real full-four117 geometric center is a consistency reference, never true COM. Real loss is synthetic feature removal, NOT event-pixel occlusion/new RAW pipeline. No new sparseBOX optimization.',
        research=[dict(url='https://robots.ox.ac.uk/~vgg/publications/1992/Rothwell92c/',scope='Known coplanar conics for pose is established, not new.'),
            dict(url='https://arxiv.org/html/2609.11920v1',scope='Section3.2 requires all four centroids in this implementation, acknowledges three sufficient in principle; known asymmetric metric geometry.'),
            dict(url='https://github.com/Zibin6/LOPET',scope='Author MATLAB model/event-line tracker; reviewed prior art, not executed or ported here.'),
            dict(url='https://link.springer.com/article/10.1186/s41074-018-0050-y',scope='Coplanar conic homography ambiguity has prior analysis; local convergence does not resolve all identities.')],
        originality='Candidate narrow contribution: learned dimensionless event-rotor structure + partial conic update + separate shape/association/pose reliability. All components have prior art. No first-ever claim, independent-method novelty unproven.',default_promoted=False))

def verify():
    p=json.loads((OUT/'protocol119.json').read_text('utf-8'));assert p['sources']==sources();return p

def baseline(rotors,body,old=None):
    if len(rotors)!=4:return dict(valid=False,reason='needs_four_current',current_count=len(rotors))
    if old is not None:return copy.deepcopy(old)
    return geom.rigid104.measure_four(rotors,body)

def banks():return {name:(PartialModel(name),CenterBank()) for name in PARAMS['modes']}

def step(models,t,body,rotors,axis,m):
    out={}
    for name,(model,bank) in models.items():
        _,d=model.step(t,body,rotors,axis,m);out[name]=dict(center=d,forecast=bank.step(t,body,d,axis))
    coeff=[o['forecast']['coefficients'].get(NAMES[0]) for o in out.values()];assert all(z==coeff[0] for z in coeff)
    return out

def run(case,mask=0,prefix=False):
    p=verify();assert p['input_sha256'][case]==sha(OLD/case/'records.jsonl')
    count=200 if case=='clear6' else 100 if case=='side6' else 30
    if prefix:count=min(count,100)
    folder=OUT/(f'{case}_mask{mask}'+('_prefix' if prefix else ''));folder.mkdir(exist_ok=False)
    models=banks();records=[]
    for i,r in enumerate(rows(OLD/case/'records.jsonl')[:count]):
        o=r['outputs']['merge_safe'];allrotors=o['geometry'].get('rotors',[]);rotors=copy.deepcopy(allrotors);removed=False
        if case=='clear6' and (65<=i<125 or 155<=i<185) and len(rotors)==4:
            centers=np.array([[z['cx'],z['cy']] for z in rotors]);order=np.argsort(np.arctan2(*(centers-centers.mean(0))[:,::-1].T))
            del rotors[int(order[mask])];removed=True
        m=baseline(rotors,r['body'],o['center']['measurement'] if not removed else None)
        out=step(models,r['t'],r['body'],rotors,r['axis'],m)
        records.append(dict(t=r['t'],index=i,body=r['body'],axis=r['axis'],input_rotors=rotors,removed_one_observation=removed,outputs=out))
    with (folder/'records.jsonl').open('x',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
    save(folder/'checks119.json',dict(frames=len(records),removed_frames=sum(r['removed_one_observation'] for r in records),
        fourth_input_absent=True,current_feature_cache=True,new_RAW=False,body_control_exact_all_modes=True,source_unchanged=verify()['sources']==sources()))
    if prefix:
        assert stable(records)==stable(rows(OUT/f'{case}_mask{mask}'/'records.jsonl')[:len(records)])
        save(folder/'prefix119.json',dict(frames=count,independent_prefix=True,all_states_forecasts_exact=True))
    print(json.dumps(dict(case=case,mask=mask,prefix=prefix,frames=len(records))),flush=True)

def synthetic():
    verify();allrecords=[]
    for case in ['stationary_wobble','translation_wobble','abrupt_turn','edge_on','two_missing']:
        models=banks();rng=np.random.default_rng(119);records=[]
        for i in range(150):
            t=i*.02;center=np.array([320.,240.])
            if case=='translation_wobble':center+=[60*t,12*t]
            if case=='abrupt_turn':center+=[65*min(t,1.4)-100*max(t-1.4,0),0.]
            height=36.+5*np.sin(3*t)
            if case=='edge_on' and 45<=i<100:height=2.
            H=np.array([[55.,5*np.sin(2*t),center[0]],[3*np.cos(t),height,center[1]],[.10*np.sin(3*t),.08*np.cos(2*t),1.]])
            full=[];allpoints=[];phase=np.linspace(0,2*np.pi,80,endpoint=False)
            for xy in np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])*[1.,1.15]:
                q=project(H,xy+.32*np.c_[np.cos(phase),np.sin(phase)])+rng.normal(0,.3,(80,2));allpoints.append(q)
                (x,y),(a,b),angle=cv2.fitEllipseDirect(np.float32(q))
                if a<b:a,b=b,a;angle+=90
                full.append(dict(cx=float(x),cy=float(y),major=float(a),minor=float(b),angle=float(angle),mode='ellipse',geometry_quality=.95,center_sigma_px=1.,inferred_shape=False))
            points=np.concatenate(allpoints);lo=points.min(0);hi=points.max(0);body=dict(detected=True,center=((lo+hi)/2).tolist(),bbox=np.r_[lo-5,hi+5].tolist(),confidence=.95)
            rotors=copy.deepcopy(full);missing=45<=i<100
            if missing:rotors=rotors[:2] if case=='two_missing' else rotors[:3]
            axis=dict(valid=case=='edge_on' and missing,angle_rad=0.,confidence=.9)
            m=baseline(rotors,body);out=step(models,t,body,rotors,axis,m)
            records.append(dict(t=t,index=i,truth=center.tolist(),body=body,input_rotors=rotors,removed_one_observation=missing and len(rotors)==3,intervention=missing,outputs=out))
        allrecords.append(dict(case=case,records=records));print(case,flush=True)
    save(OUT/'synthetic119.json',dict(cases=allrecords,scope='Noisy ideal projective circular rims; not native-event occlusion or true rotor physics. Real motion and wobble independent.'))

def evaluate():
    verify();table=[];refs=rows(REFERENCE/'clear6/records.jsonl')
    for mask in range(4):
        rr=rows(OUT/f'clear6_mask{mask}'/'records.jsonl')
        for mode in PARAMS['modes']:
            distance=[];futures=[];reason=collections.Counter();current=0;eligible=0
            for i,r in enumerate(rr):
                if not r['removed_one_observation']:continue
                d=r['outputs'][mode]['center'];ref=refs[i]['outputs']['fixed_structure']['center'];reason[d['measurement']['reason']]+=1
                current+=int(d['current']);eligible+=1
                if d.get('center') is not None and ref['current']:distance.append(float(np.linalg.norm(np.array(d['center'])-ref['center'])))
                if i+20<len(rr):
                    c=r['outputs'][mode]['forecast']['coefficients'].get(NAMES[2]);target=refs[i+20]['outputs']['fixed_structure']['forecast']['coefficients'].get(NAMES[2])
                    if c and target:futures.append(float(np.linalg.norm(np.array(c['position'])+.4*np.array(c['velocity'])-target['position'])))
            table.append(dict(case=f'clear_mask{mask}',mode=mode,removed_frames=eligible,current_geometry=current,
                center_consistency_rmse_px=float(np.sqrt(np.mean(np.square(distance)))) if distance else None,center_reference_pairs=len(distance),
                reference_fde04_mean_px=float(np.mean(futures)) if futures else None,reference_forecast_pairs=len(futures),reasons=dict(reason)))
    synthetic=[]
    for item in json.loads((OUT/'synthetic119.json').read_text('utf-8'))['cases']:
        rr=item['records']
        for mode in PARAMS['modes']:
            first=next((r for r in rr if r['outputs'][mode]['center']['current']),None)
            offset=np.array(first['outputs'][mode]['forecast']['coefficients'][NAMES[2]]['position'])-first['truth'] if first else np.zeros(2)
            errors=[];futures=[];current=0;reasons=collections.Counter()
            for i,r in enumerate(rr):
                if not r['intervention']:continue
                d=r['outputs'][mode]['center'];current+=int(d['current']);reasons[d['measurement']['reason']]+=1
                c=r['outputs'][mode]['forecast']['coefficients'].get(NAMES[2])
                if not c:continue
                errors.append(float(np.linalg.norm(np.array(c['position'])-r['truth']-offset)))
                if i+20<len(rr):futures.append(float(np.linalg.norm(np.array(c['position'])+.4*np.array(c['velocity'])-rr[i+20]['truth']-offset)))
            synthetic.append(dict(case=item['case'],mode=mode,current_geometry=current,intervention_frames=55,
                relative_position_rmse_px=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                fde04_mean_px=float(np.mean(futures)) if futures else None,reasons=dict(reasons)))
    save(OUT/'scores119.json',dict(real_consistency=table,synthetic=synthetic,real_COM_ground_truth=False,new_RAW=False,physical_occlusion_tested=False))
    print(json.dumps(dict(real=table,synthetic=synthetic)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','prefix','synthetic','evaluate']);p.add_argument('--case',choices=['clear6','side6','negative5'],default='clear6');p.add_argument('--mask',type=int,choices=range(4),default=0);a=p.parse_args()
    if a.action in ['freeze','synthetic','evaluate']:globals()[a.action]()
    else:run(a.case,a.mask,a.action=='prefix')
