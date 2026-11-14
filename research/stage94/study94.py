"""Current official194 native-HDF5 past-box feasibility/forecast diagnostic."""
from pathlib import Path
import argparse,copy,datetime,inspect,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage88'),str(ROOT/'research/stage89'),str(ROOT/'research/stage93'),str(ROOT/'research/stage92'),str(ROOT/'research/stage82'),str(ROOT/'research/stage6'),str(ROOT/'research/stage79')]
from inventory88 import save,load,sha,parse,unique_rows
from paired88 import Paired,prior
from stream82 import StreamingActivity
from silhouette import body_measurement
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.rotor_sensing import PaperPerception
from event_drone_forecast.paper_physics import ThrustAxisFilter
from event_drone_forecast.raw_io import raw_chunks
from online93 import Kinematic
from router92 import Router92
from hist89 import PeriodHistogram
from native79 import engine,checked_process,summarize
import score88
OUT=ROOT.parent/'output/stage94_official_large_target_20261022'
METHODS=['cv','trusted_acc','event','constant','zero','lag200','past_shuffle101','past_shuffle202','past_shuffle303','kf_tuned','fallback_router','full_router']


def sources():
    return {str(p):sha(p) for p in [Path(__file__),ROOT/'research/stage88/paired88.py',ROOT/'research/stage89/hist89.py',
        ROOT/'research/stage93/online93.py',ROOT/'research/stage92/router92.py',ROOT/'research/stage79/native79.py',ROOT/'research/stage88/score88.py']}


def freeze():
    assert not (OUT/'protocol_feature94.json').exists();data=load(OUT/'protocol94.json')
    save(OUT/'protocol_feature94.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        data_protocol_sha256=sha(OUT/'protocol94.json'),start_s=data['start_s'],duration_s=10.,step_us=20000,
        design='Same88 past-box/raw count geometry, body and160ms frequency parameters; same72/77/84 physics and matched controls; same90 sigma6400 and92 router.',
        native_reader='Installed Prophesee HDF5EventsReader (native event format); no RAW conversion, no RGB/G input.',
        interpretation='New size-selected sequence194 from current official revision. 10s midpoint window fixed before downloading events; metadata selection does not make unbiased benchmark.',
        candidate_frequencies='89 guarded/raw histogram and native strict7 are diagnostic only; never fabricated as physical frequencies.',
        scoring='88 actual-label ADE/endpoint rules, regular100ms anchors from start+.6 through end-.4, .2/.4 horizons, no interpolation.',
        actual_event_stream_prefix_s=1.,mechanical_rpm_truth=False,RAW_only_deployment=False,default_promoted=False))
    print('94 native large-target diagnostic frozen before event inspection',flush=True)


def verify():
    p=load(OUT/'protocol_feature94.json');assert p['source_hashes']==sources();assert p['data_protocol_sha256']==sha(OUT/'protocol94.json');return p


def run(prefix=False):
    p=verify();download=load(OUT/'download94.json');raw=Path(download['path']);assert sha(raw)==download['native_sha256']
    from metavision_core.event_io.h5_io import HDF5EventsReader
    start=p['start_s'];duration=1. if prefix else p['duration_s'];folder=OUT/'fred'/('prefix194' if prefix else 'sequence194');folder.mkdir(parents=True,exist_ok=False)
    annotation=OUT/'input/coordinates.txt';labels,bad=parse(annotation.read_text(encoding='utf-8-sig'));assert not bad;labels=unique_rows(labels)
    if prefix:labels=[r for r in labels if r['t']<=start+duration]
    reader=HDF5EventsReader(str(raw));height,width=reader.get_size();cfg=PaperConfig();cv2.setNumThreads(1)
    geom=PaperPerception(width,height,cfg);axis=ThrustAxisFilter(cfg);activity=StreamingActivity(width,height)
    motion=prior.robust.Predictor();paired=Paired();kinematic=Kinematic();router=Router92();hist=PeriodHistogram(width,height)
    native=engine(width,height,'strict7');native_output=native.get_empty_output_buffer()
    index=0;latest=None;last_id=None;last_label_t=None;count=0;timing=[];frames=[];beg=time.perf_counter();last_progress=beg
    counters=dict(events=0,position_updates=0,rotor_geometry_ticks=0,body_axis_ticks=0,qualified_spectral_ticks=0,
        guarded_histogram_ticks=0,raw_histogram_ticks=0,native_strict_ticks=0,current_physics_ticks=0)
    writer=None
    if not prefix:
        writer=cv2.VideoWriter(str(folder/'native_predictions.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10.,(width,height));assert writer.isOpened()
    with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as f:
        for t,e in raw_chunks(reader,start,duration,20000):
            started=time.perf_counter();new=None
            while index<len(labels) and labels[index]['t']<=t+1e-9:
                r=labels[index];index+=1;latest=r
                if r['t']>=start-1e-9:new=r
            b=None if latest is None or t-latest['t']>.05+1e-9 else latest['bbox']
            if new is not None:
                changed=last_id is not None and new['identity'] is not None and new['identity']!=last_id
                if changed or (last_label_t is not None and new['t']-last_label_t>.5):
                    geom=PaperPerception(width,height,cfg);axis=ThrustAxisFilter(cfg);paired=Paired();kinematic=Kinematic();router=Router92();motion=prior.robust.Predictor()
                last_id=new['identity'];last_label_t=new['t']
            z=None if new is None else ((np.array(new['bbox'][:2])+new['bbox'][2:])/2).tolist()
            obs=dict(detected=z is not None,center=z,confidence=1. if z is not None else 0.)
            q,_=motion.step(t,z);v=np.array(q['coefficients']['cv']['velocity']) if q['valid'] else np.zeros(2)
            cropped=e[:0] if b is None else e[(e['x']>=max(0,b[0]))&(e['x']<min(width,b[2]))&(e['y']>=max(0,b[1]))&(e['y']<min(height,b[3]))]
            rotor,_=geom.detect(cropped,t);normal=axis.step(t,rotor.get('rotors',[]),0.,True)
            age=None if axis.last_measurement_t is None else t-axis.last_measurement_t
            q['feature_diagnostic']=dict(axis_age_s=age,models={'adaptive_axis_only':{'normal':None if normal is None else normal.tolist()}})
            density=np.bincount(e['y'].astype(int)*width+e['x'],minlength=width*height).reshape(height,width).astype(np.float32)
            body=dict(valid=False,reason='no_ROI') if b is None else body_measurement(density,b)
            a=activity.step(e,b,t,v);result=paired.step(q,obs,body,a,axis.quality)
            k=kinematic.step(t,z);kin=dict(t=t,valid=q['valid'],coefficients={})
            if q['valid']:kin['coefficients']=dict(cv=q['coefficients']['cv'],trusted_acc=q['coefficients']['trusted_acc'],kf_tuned=k)
            routed=router.step(result,kin,obs)
            if q['valid']:
                result['coefficients'].update(kf_tuned=k,fallback_router=routed['coefficients']['fallback_router'],full_router=routed['coefficients']['full_router'])
            hd=hist.step(e,b,t);nd=summarize(checked_process(native,e,native_output,width,height),b,t,width,height)
            ms=(time.perf_counter()-started)*1000;timing.append(ms)
            result['input_audit']=dict(bbox=b,past_label_t=None if b is None else latest['t'],current_label_t=None if new is None else new['t'],
                event_count=len(e),geometry_rotors=len(rotor.get('rotors',[])),body_valid=body.get('valid',False),
                qualified_spectral_candidates=sum(c['qualified'] for c in a.get('spectral_candidates',[])),histogram=hd,native=nd,processing_ms=ms)
            f.write(json.dumps(result,allow_nan=False)+'\n');count+=1
            counters['events']+=len(e);counters['position_updates']+=int(z is not None);counters['rotor_geometry_ticks']+=bool(rotor.get('rotors'))
            counters['body_axis_ticks']+=bool(body.get('valid'));counters['qualified_spectral_ticks']+=bool(a.get('frequency_squared'))
            counters['guarded_histogram_ticks']+=hd['guarded_qualified'];counters['raw_histogram_ticks']+=hd['raw_qualified'];counters['native_strict_ticks']+=nd['qualified']
            counters['current_physics_ticks']+=result['diagnostics']['fresh']
            if writer is not None and count%5==0:
                image=cv2.cvtColor(np.uint8(np.clip(np.log1p(density)/4,0,1)*255),cv2.COLOR_GRAY2BGR)
                if b is not None:cv2.rectangle(image,tuple(np.int32(b[:2])),tuple(np.int32(b[2:])),(0,200,0),1)
                if result['valid']:
                    for name,color in [('cv',(255,130,0)),('event',(0,220,255)),('fallback_router',(255,255,0)),('kf_tuned',(0,130,255))]:
                        xy=np.array([prior.evaluate(result['coefficients'][name],h) for h in [.1,.2,.4]])
                        if np.isfinite(xy).all() and abs(xy).max()<100000:cv2.polylines(image,[np.int32(xy)],False,color,2);cv2.circle(image,tuple(np.int32(xy[-1])),5,color,-1)
                cv2.putText(image,f'Official FRED194 t={t:.2f} | past-box diagnostic | physics={int(result["diagnostics"]["fresh"])}',(12,24),0,.55,(230,230,230),1)
                cv2.putText(image,'blue=CV yellow=periodic primary cyan=fallback orange=train-KF',(12,height-18),0,.55,(255,255,0),1)
                writer.write(image)
                if count%50==0 and len(frames)<9:frames.append(cv2.resize(image,(640,360)))
            if time.perf_counter()-last_progress>15:
                print('94 native events',t,counters,flush=True);last_progress=time.perf_counter()
    if writer is not None:
        writer.release();cap=cv2.VideoCapture(str(folder/'native_predictions.mp4'));decoded=0
        while True:
            ok,image=cap.read()
            if not ok:break
            decoded+=1
        cap.release();assert decoded==count//5
    if len(frames)>=9:cv2.imwrite(str(folder/'fixed_contact_sheet.png'),np.vstack([np.hstack(frames[i:i+3]) for i in [0,3,6]]))
    if prefix:
        old=score88.rows(OUT/'fred/sequence194/forecasts.jsonl')[:count];new=score88.rows(folder/'forecasts.jsonl')
        for a,b in zip(old,new):a['input_audit'].pop('processing_ms');b['input_audit'].pop('processing_ms');assert a==b
    save(folder/'completion.json',dict(sequence=194,ticks=count,**counters,actual_native_HDF5=True,past_provided_boxes=True,
        RAW_only_deployment=False,source_hashes_unchanged=p['source_hashes']==sources(),actual_event_prefix_exact=prefix,
        module_median_ms=float(np.median(timing)),module_p95_ms=float(np.percentile(timing,95)),wall_s=time.perf_counter()-beg))
    print('94 completed',prefix,counters,flush=True)


def score():
    p=verify();assert not (OUT/'scores94.json').exists();assert load(OUT/'fred/prefix194/completion.json')['actual_event_prefix_exact']
    (OUT/'annotations').mkdir(exist_ok=True);(OUT/'annotations/194.txt').write_bytes((OUT/'input/coordinates.txt').read_bytes())
    source=inspect.getsource(score88.fred_score)
    old='t<20.6-1e-8 or t>49.6+1e-8';new=f't<{p["start_s"]+.6}-1e-8 or t>{p["start_s"]+9.6}+1e-8'
    assert source.count(old)==1;source=source.replace(old,new)
    namespace=dict(score88.fred_score.__globals__,OUT=OUT,METHODS=METHODS)
    exec(compile(source,'<94 unchanged88 scoring with declared midpoint interval>','exec'),namespace)
    protocol=load(ROOT.parent/'output/stage88_causal_event_controls_20261022/protocol88.json');protocol['fred_cases']=[194]
    details,audit=namespace['fred_score'](protocol);table=score88.summarise(details)
    save(OUT/'endpoint_details94.json',details);save(OUT/'scores94.json',dict(table=table,audit=audit,
        data_size_conditioned=True,same_parameters_as88_90_92=True,independent_mechanical_RPM_truth=False,default_promoted=False))
    print('94 actual native large-target scores',[(r['method'],r['valid'],r['mean_fde']) for r in table if r['horizon']==.4 and r['group']=='all'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','prefix','score']);a=p.parse_args()
    if a.action=='prefix':run(True)
    else:globals()[a.action]()
