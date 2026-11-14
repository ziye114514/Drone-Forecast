"""Frozen persistent learned-structure versus per-frame free-shape controls."""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage117'),str(ROOT/'research/stage113'),str(ROOT/'research/stage110')]
from fixed117 import FixedModel,fit_fixed,PARAMS,geom,DERIVED_FIXED_SOURCE
from study113 import stable,CenterBank,NAMES,SUPPORT
from study110 import OUT as OLD,CASES
from audit109 import rows,save,sha
OUT=ROOT.parent/'output/stage117_persistent_structure_20261022'
def sources():return {p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),ROOT/'research/stage117/fixed117.py',ROOT/'research/stage113/support113.py',ROOT/'research/stage104/rigid104.py',ROOT/'research/stage107/center107.py']}
def freeze():
    assert not OUT.exists();save(OUT/'protocol117.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),parameters=PARAMS,
        input_sha256={c:sha(OLD/c/'records.jsonl') for c in CASES},
        derived_sha256=hashlib.sha256(DERIVED_FIXED_SOURCE.encode()).hexdigest(),
        design='Original same-current110 geometry and full-perimeter objective; three branches free structure, five-frame learned fixed structure, same plus conditional pose uncertainty. Fixed dimensionless shape retained separately from current pose while body identity continuous; original pose/5-frame/fallback rules unchanged.',
        reason='104 learns only radius/sqrt(aspect) for post-fit rejection and frees radius/aspect in every fit. Implement actual fixed shape only after unchanged qualifying initialization. No importing22s pose or structural calibration into110s; each case independent.',
        controls='Synthetic known projective circle sequence: stationary wobble/real translation/abrupt turn/temporary missing rotors. Same0.3px rim noise and original admission, not native-event physics. No stationary prior; record failures.',
        evaluation='Previously examined sparseBOX endpoints, only read after all forecasts. Synthetic center truth separately known, real COM/RPM/pose truth unavailable. No current per-rotor RPM claim or promotion.',default_promoted=False))
    (OUT/'derived_fixed_fit117.py').write_text(DERIVED_FIXED_SOURCE,encoding='utf-8')
def verify():
    p=json.loads((OUT/'protocol117.json').read_text('utf-8'));assert p['sources']==sources();assert p['derived_sha256']==hashlib.sha256(DERIVED_FIXED_SOURCE.encode()).hexdigest();return p

def run(case,prefix=False):
    p=verify();assert p['input_sha256'][case]==sha(OLD/case/'records.jsonl');rr=rows(OLD/case/'records.jsonl');rr=rr[:60] if prefix else rr
    folder=OUT/(case+('_prefix' if prefix else ''));folder.mkdir(exist_ok=False)
    models={'free_structure':geom.MeasuredCenter(),'fixed_structure':FixedModel(),'fixed_guarded':FixedModel(True)};banks={m:CenterBank() for m in PARAMS['modes']};records=[]
    with (folder/'records.jsonl').open('x',encoding='utf-8') as stream:
        for r in rr:
            o=r['outputs']['merge_safe'];rotors=o['geometry'].get('rotors',[]);oldm=o['center']['measurement'];fit=oldm.get('fit')
            baseline=geom.measure_fitted(rotors,r['body'],fit) if fit is not None else copy.deepcopy(oldm);outputs={}
            for mode,model in models.items():
                if mode=='free_structure':model.input_measurement=baseline;obs,d=model.step(r['t'],r['body'],rotors,r['axis'])
                else:obs,d=model.step(r['t'],r['body'],rotors,r['axis'],baseline)
                q=banks[mode].step(r['t'],r['body'],d,r['axis']);outputs[mode]=dict(center=d,forecast=q)
            assert outputs['free_structure']['forecast']==o['forecast'] and stable(outputs['free_structure']['center'])==stable(o['center'])
            body=[z['forecast']['coefficients'].get(NAMES[0]) for z in outputs.values()];assert all(z==body[0] for z in body)
            q=dict(t=r['t'],body=r['body'],axis=r['axis'],valid=r['valid'],outputs=outputs);stream.write(json.dumps(q,allow_nan=False)+'\n');records.append(q)
            if len(records)%100==0:print(f'{case} fixed-shape {len(records)}',flush=True)
    save(folder/'checks117.json',dict(frames=len(records),current_geometry_cache=True,new_RAW=False,free_structure_exact110=True,bodyKF_all_modes_exact=True,source_unchanged=verify()['sources']==sources(),labels_read=False))
    if prefix:
        assert stable(records)==stable(rows(OUT/case/'records.jsonl')[:len(records)])
        save(folder/'prefix117.json',dict(independent_prefix=True,frames=len(records),all_states_and_forecasts_exact=True))
    print(json.dumps(dict(case=case,prefix=prefix,frames=len(records))),flush=True)

def synthetic():
    verify();rng=np.random.default_rng(117);allrecords=[];table=[]
    for case in ['stationary_wobble','translation_wobble','abrupt_turn','missing_and_recovery']:
        free=geom.MeasuredCenter();fixed=FixedModel();banks={m:CenterBank() for m in ['free','fixed']};records=[]
        for i in range(100):
            t=i*.02;center=np.array([320.,240.])
            if case=='translation_wobble':center+=np.array([70.*t,15.*t])
            if case=='abrupt_turn':center+=np.array([70.*min(t,.9)-100.*max(t-.9,0),0.])
            H=np.array([[55.,5.*np.sin(2*t),center[0]],[3.*np.cos(t),36.+4.*np.sin(3*t),center[1]],[.10*np.sin(3*t),.08*np.cos(2*t),1.]])
            world=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])*[1.,1.15];rotors=[];boundary=[]
            for xy in world:
                phase=np.linspace(0,2*np.pi,80,endpoint=False);uv=xy+.32*np.c_[np.cos(phase),np.sin(phase)]
                hp=np.c_[uv,np.ones(len(uv))]@H.T;p=hp[:,:2]/hp[:,2,None]+rng.normal(0,.3,(len(uv),2));boundary.append(p)
                (x,y),(a,b),angle=cv2.fitEllipseDirect(np.float32(p))
                if a<b:a,b=b,a;angle+=90
                rotors.append(dict(cx=float(x),cy=float(y),major=float(a),minor=float(b),angle=float(angle),mode='ellipse',geometry_quality=.95,center_sigma_px=1.,inferred_shape=False))
            p=np.concatenate(boundary);lo=p.min(0);hi=p.max(0);body=dict(detected=True,center=((lo+hi)/2).tolist(),bbox=np.r_[lo-5,hi+5].tolist(),confidence=.95)
            missing=case=='missing_and_recovery' and 30<=i<65
            if missing:rotors=rotors[:2]
            baseline=geom.rigid104.measure_four(rotors,body);axis=dict(valid=missing,angle_rad=0.,confidence=.9)
            free.input_measurement=baseline;_,fd=free.step(t,body,rotors,axis);_,sd=fixed.step(t,body,rotors,axis,baseline)
            out={}
            for name,d in [('free',fd),('fixed',sd)]:out[name]=dict(center=d,forecast=banks[name].step(t,body,d,axis))
            records.append(dict(t=t,truth=center.tolist(),body=body,rotors=rotors,axis=axis,baseline=baseline,outputs=out))
        for name in ['free','fixed']:
            # Gauge align once at first published geometry; this is relative
            # motion truth, not a hidden real-camera COM calibration.
            first=next((r for r in records if r['outputs'][name]['center']['current']),None);errors=[];future=[]
            offset=np.asarray(first['outputs'][name]['forecast']['coefficients'][NAMES[2]]['position'])-first['truth'] if first else np.zeros(2)
            for i,r in enumerate(records):
                if i<20:continue
                c=r['outputs'][name]['forecast']['coefficients'].get(NAMES[2])
                if c is None:continue
                errors.append(np.linalg.norm(np.asarray(c['position'])-r['truth']-offset))
                if i+20<len(records):future.append(np.linalg.norm(np.asarray(c['position'])+.4*np.asarray(c['velocity'])-records[i+20]['truth']-offset))
            table.append(dict(case=case,mode=name,position_relative_rmse_px=float(np.sqrt(np.mean(np.square(errors)))),fde04_mean_px=float(np.mean(future)),
                current=sum(r['outputs'][name]['center']['current'] for r in records),initializations=sum(r['outputs'][name]['center']['initialized_now'] for r in records)))
        allrecords.append(dict(case=case,records=records))
    save(OUT/'synthetic117.json',dict(table=table,records=allrecords,scope='Known projective circular perimeters with.3px noise; not native-event rendering or realCOM truth. First-current gauge alignment only for relative displacement score.'))
    print(json.dumps(table),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','prefix','synthetic']);p.add_argument('--case',choices=CASES);a=p.parse_args()
    if a.action in ['freeze','synthetic']:globals()[a.action]()
    else:run(a.case,a.action=='prefix')
