"""Fixed full-window RAW test of native weak-support recovery, with controls."""
from pathlib import Path
import argparse,collections,copy,datetime,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage114'),str(ROOT/'research/stage113'),str(ROOT/'research/stage112'),str(ROOT/'research/stage110')]
from weak114 import recover,PARAMS as RECOVERY_PARAMS
from capture112 import SupportFrontend,key,OUT as SUPPORT
from study113 import geom,stable,CenterBank,NAMES
from study110 import CASES,OUT as OLD
from audit109 import save,sha,rows
from windows21 import Observer
from event_drone_forecast.raw_io import open_raw,raw_chunks
OUT=ROOT.parent/'output/stage115_RAW_weak_support_20261022'
PARAMS=dict(modes=['strong_perimeter','weak_contour','weak_guarded'],recovery=RECOVERY_PARAMS,
    cropping='Current measured rotor extent with.85*major circular padding plus3px, clipped to sensor. Density threshold computed on full current frame exactly as110; crop does not change threshold.',
    reason='114 C regained80-budget conic fit after half-threshold recovery, but clear sample optimizer failed. Freeze full clear/side/continuous/negative comparison before inspecting their weak-boundary predictions.',
    scope='Raw geometric frontend with cached contemporaneous body/axis. Conditional branch not default; no mechanical RPM, pose truth, independent recording or complete new RAW-only deployment claim.')

class ActivitySupport(SupportFrontend):
    def _candidates(self,contours,activity,*args,**kwargs):
        self.current_activity=activity
        return super()._candidates(contours,activity,*args,**kwargs)

def sources():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),ROOT/'research/stage114/weak114.py',ROOT/'research/stage113/support113.py',
        ROOT/'research/stage112/capture112.py',ROOT/'research/stage110/merge110.py',ROOT/'research/stage107/center107.py',ROOT/'research/stage104/rigid104.py']}

def freeze():
    assert not OUT.exists();save(OUT/'protocol115.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),parameters=PARAMS,cases=CASES,
        reference_sha256={c:sha(OLD/c/'records.jsonl') for c in CASES},
        design='Re-read RAW, exactly match old merge-safe geometry before recovery. Same three independent104 model states/107 banks; no missing-point completion, no gate reduction. Same original body/axis per time for controls.',
        evaluation='All fixed windows retained; existing sparse BOX future targets only loaded after forecasts.114 reviewed snapshots are development and can influence choice of this hypothesis; not blind.',
        tests='True RAW truncated prefixes with all states/forecasts exact except timings. Original110 model/forecast exact and body-onlyKF exact across branches; negative case.',default_promoted=False))

def verify():
    p=json.loads((OUT/'protocol115.json').read_text('utf-8'));assert p['sources']==sources();return p

def run(case,prefix=False):
    p=verify();cfg=CASES[case];folder=OUT/(case+('_prefix' if prefix else ''));folder.mkdir(exist_ok=False)
    assert p['reference_sha256'][case]==sha(OLD/case/'records.jsonl')
    refs={round(r['t'],8):r for r in rows(OLD/case/'records.jsonl')}
    reader=open_raw(cfg['raw']);h,w=reader.get_size();front=Observer(w,h,'native20');front.geometry=ActivitySupport(w,h)
    models={mode:(geom.MeasuredCenter(),CenterBank()) for mode in PARAMS['modes']}
    begin=max(0.,cfg['start']-.1);duration=min(cfg['duration'],1.2) if prefix else cfg['duration'];records=[]
    with (folder/'records.jsonl').open('x',encoding='utf-8') as stream:
        for t,e in raw_chunks(reader,begin,duration+cfg['start']-begin,20000):
            obs=front.step(e,t)['observation'] if t-begin>=.1-1e-8 else dict(t=t,detected=False,rotors=[])
            if t<=cfg['start']+1e-8:continue
            r=refs[round(t,8)];old=r['outputs']['merge_safe'];assert obs==old['geometry'],(case,t,'input geometry changed')
            rotors=obs.get('rotors',[]);oldm=old['center']['measurement'];fit=oldm.get('fit');baseline=geom.measure_fitted(rotors,r['body'],fit) if fit else copy.deepcopy(oldm)
            measures={'strong_perimeter':baseline};recovery=dict(reason='no_four_supported_seeds');weakrotors=[];weakpoints=[]
            support=[front.geometry.supports.get(key(z),[]) for z in rotors]
            if fit is not None and len(rotors)==4 and all(len(z)>=6 for z in support):
                points=np.array([[q['cx'],q['cy']] for q in rotors]);radius=np.array([q['major']*.85+3 for q in rotors])[:,None]
                lo=np.maximum(np.floor(np.min(points-radius,axis=0)).astype(int),0);hi=np.minimum(np.ceil(np.max(points+radius,axis=0)).astype(int)+1,[w,h])
                x0,y0=lo;x1,y1=hi
                density=cv2.GaussianBlur(front.geometry.current_activity,(3,3),.8)[y0:y1,x0:x1]
                threshold=front.geometry.merge_audit['threshold']
                recovered,recovery,_=recover(density,threshold,lo,rotors,support)
                weakrotors=[v[0] for v in recovered];weakpoints=[v[1] for v in recovered]
                if len(recovered)==4:
                    used=geom.supports_for(weakpoints);wf=geom.support_fit([geom.geometry58.from_rotor(z) for z in weakrotors],used)
                    if wf.get('valid'):wf['support_coverage']=geom.coverage(wf,used)
                    m=geom.measure_fitted(weakrotors,r['body'],wf);measures['weak_contour']=m;measures['weak_guarded']=geom.guarded(m)
            if 'weak_contour' not in measures:
                m=dict(valid=False,reason='weak_support_incomplete',current_count=len(weakrotors))
                measures['weak_contour']=m;measures['weak_guarded']=copy.deepcopy(m)
            outputs={}
            for mode,(center,bank) in models.items():
                center.input_measurement=measures[mode];used_rotors=rotors if mode=='strong_perimeter' else weakrotors
                _,d=center.step(t,r['body'],used_rotors,r['axis']);q=bank.step(t,r['body'],d,r['axis']);outputs[mode]=dict(center=d,forecast=q)
            assert outputs['strong_perimeter']['forecast']==old['forecast'] and stable(outputs['strong_perimeter']['center'])==stable(old['center'])
            body=[v['forecast']['coefficients'].get(NAMES[0]) for v in outputs.values()];assert all(z==body[0] for z in body)
            q=dict(t=t,body=r['body'],axis=r['axis'],valid=r['valid'],outputs=outputs,recovery=recovery)
            if abs(t-112.02)<1e-8:q['fixed_C_recovered_rotors']=weakrotors;q['fixed_C_recovered_points']=[z.tolist() for z in weakpoints]
            stream.write(json.dumps(q,allow_nan=False)+'\n');records.append(q)
            if len(records)%100==0:print(f'{case} weak-support {len(records)}',flush=True)
    assert len(records)==round(duration*50)
    save(folder/'checks115.json',dict(frames=len(records),actual_RAW_geometry=True,cached_current_body=True,
        strong_frontend_exact110=True,strong_model_and_forecast_exact110=True,bodyKF_all_modes_exact=True,labels_read=False,source_unchanged=verify()['sources']==sources()))
    if prefix:
        full=rows(OUT/case/'records.jsonl')[:len(records)];assert stable(full)==stable(records)
        save(folder/'prefix115.json',dict(actual_truncated_RAW=True,frames=len(records),all_states_and_forecasts_exact=True))
    print(json.dumps(dict(case=case,prefix=prefix,frames=len(records))),flush=True)

def summarize():
    verify();summary=[]
    for case in CASES:
        rr=rows(OUT/case/'records.jsonl')
        for mode in PARAMS['modes']:
            ds=[r['outputs'][mode]['center'] for r in rr];summary.append(dict(case=case,mode=mode,frames=len(rr),
                qualified=sum(r['measurement']['valid'] for r in ds),current_centers=sum(r['current'] for r in ds),initializations=sum(r['initialized_now'] for r in ds),
                reasons=dict(collections.Counter(r['measurement']['reason'] for r in ds))))
    save(OUT/'availability115.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','prefix','summarize']);p.add_argument('--case',choices=CASES);a=p.parse_args()
    if a.action in ['freeze','summarize']:globals()[a.action]()
    else:run(a.case,a.action=='prefix')
