"""Frozen diagnostic: existing auto-RAW caches plus fresh FRED past-box/RAW inputs."""
from pathlib import Path
import argparse, collections, copy, datetime, hashlib, json, sys, time
import cv2, numpy as np
from inventory88 import ROOT, OUT, OLD, save, load, sha, parse, unique_rows
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT/'research/stage82'),str(ROOT/'research/stage6')]
from paired88 import Paired, METHODS, PARAMS, controls, minimal_q, prior
from stream82 import StreamingActivity
from silhouette import body_measurement
from event_drone_forecast.raw_io import open_raw,raw_chunks
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.rotor_sensing import PaperPerception
from event_drone_forecast.paper_physics import ThrustAxisFilter

SELF=ROOT.parent/'output/stage78_RAW_separated_forecast_20261021'
SELF_CASES=['clear6','side6','train3','extra1','extra2','extra4']
FRED_CASES=[0,4,6,10,8,9]


def sources():
    files=[Path(__file__),Path(__file__).with_name('paired88.py'),Path(__file__).with_name('inventory88.py'),Path(__file__).with_name('RELATED_WORK.md')]
    files+=list((ROOT/'src/event_drone_forecast').glob('*.py'))
    for stage in [6,14,22,23,27,62,64,66,68,69,72,73,74,77,78,80,82,84]:
        files+=list((ROOT/f'research/stage{stage}').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def freeze():
    assert not (OUT/'protocol88.json').exists()
    inv=load(OUT/'inventory88.json')
    save(OUT/'protocol88.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        inventory_sha256=sha(OUT/'inventory88.json'),parameters=PARAMS,controls=controls(),
        self_cases=SELF_CASES,self_cache_sha256={c:sha(SELF/c/'observations_forecasts.jsonl') for c in SELF_CASES},
        fred_cases=FRED_CASES,fred_interval_s=[20.,50.],fred_decode_step_us=20000,fred_roi_max_age_s=.05,
        fred_role='Prediction-layer diagnostic using past provided positions/ROIs plus actual RAW, not RAW-only deployment. '
            'Official-test sequence8/9 were inspected in earlier stages; new interval does not make recording blind.',
        self_role='Recompute physical/MMAE controls from stored automatic78 inputs. Exact event/constant regression required; not fresh RAW run.',
        labels='Original coordinates only. Duplicate timestamps entirely dropped; no interpolation or invented IDs. '
            'Only new past labels are position updates, quantized to next20ms tick, age recorded. No later box in inference.',
        feature='Existing RAW PaperPerception inside past50ms ROI, ThrustAxisFilter, body_measurement on current counts, '
            'StreamingActivity160ms, no G/RGB input. All branches share geometry/position/gates.',
        scoring=dict(horizons=[.2,.4],anchor_stride_s=.1,endpoint_match_tolerance_s=.018,max_truth_gap_s=.101,
            minimum_elapsed_s=.6,turn_angle_deg=30.,turn_min_speed_px_s=10.,
            groups='Independent provided-box before/after velocity direction, post-hoc scoring only; no maneuver truth in predictor.',
            ADE='Mean displacement over actual provided future labels, no interpolated dense points; not official complete FRED.',
            bootstrap='1s non-overlapping anchor blocks within each sequence, 1000 resamples, seed8800; diagnostic only with two test recordings.'),
        decision='Event independent gain requires >5% lower mean0.4 endpoint error versus constant/zero/lag/median3shuffle, '
            'no worse on either fixed test recording, >=20 turn anchors. This is a pilot decision criterion, not a universal significance rule.',
        no_tuning_on_test=True,default_promoted=False))
    print('88 protocol frozen; all controls and held parameters fixed',flush=True)


def verify():
    p=load(OUT/'protocol88.json');assert p['source_hashes']==sources(),'Frozen88 source changed'
    assert p['inventory_sha256']==sha(OUT/'inventory88.json');return p


def self_run(case):
    p=verify();path=SELF/case/'observations_forecasts.jsonl';assert sha(path)==p['self_cache_sha256'][case]
    dest=OUT/'self'/case;dest.mkdir(parents=True,exist_ok=False)
    model=Paired();n=0;active=0;beg=time.perf_counter();prefix=[]
    with (dest/'forecasts.jsonl').open('x',encoding='utf-8') as f:
        for line in path.open(encoding='utf-8'):
            r=json.loads(line);q=r['frontends']['delayed_feedback'];quality=q['increment_diagnostic']['direction']['quality']
            v=model.step(minimal_q(q),r['observation'],r['body_axis'],r['event_activity'],quality)
            assert v['diagnostics']['direction']==q['increment_diagnostic']['direction'],(case,r['t'],'direction')
            assert v['diagnostics']['fresh']==q['separated_diagnostic']['current'],(case,r['t'],'clock')
            if q['valid']:
                assert v['coefficients']['event']==q['coefficients']['mmae_state_fresh'],(case,r['t'],'event')
                assert v['coefficients']['constant']==q['coefficients']['mmae_state_fresh_constant'],(case,r['t'],'constant')
            f.write(json.dumps(v,allow_nan=False)+'\n');n+=1;active+=v['diagnostics']['fresh']
            if n<=100:prefix.append(v)
    # Actual cache prefix replay is independent of future suffix.
    m=Paired();short=[]
    with path.open(encoding='utf-8') as f:
        for j,line in enumerate(f):
            if j>=len(prefix):break
            r=json.loads(line);q=r['frontends']['delayed_feedback']
            short.append(m.step(minimal_q(q),r['observation'],r['body_axis'],r['event_activity'],q['increment_diagnostic']['direction']['quality']))
    assert short==prefix
    result=dict(case=case,ticks=n,current_physics=active,event_and_constant_exact78=True,actual_cache_prefix_exact=True,
        wall_s=time.perf_counter()-beg,cache_sha256=sha(path),fresh_RAW=False)
    save(dest/'completion.json',result);print(json.dumps(result),flush=True)


def fred_run(seq,prefix_duration=None):
    p=verify();cfg=next(r for r in load(OUT/'inventory88.json')['sequences'] if r['sequence']==seq)
    raw=Path(cfg['raw']);assert sha(raw)==cfg['raw_sha256']
    all_labels,bad=parse((OUT/'annotations'/f'{seq}.txt').read_text(encoding='utf-8-sig'));assert not bad
    labels=unique_rows(all_labels)
    duration=30. if prefix_duration is None else prefix_duration
    if prefix_duration is not None:labels=[r for r in labels if r['t']<=20+duration]
    dest=OUT/'fred'/(f'sequence{seq}' if prefix_duration is None else f'prefix{seq}')
    dest.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);reader=open_raw(raw);height,width=reader.get_size()
    cfg_cam=PaperConfig();geometry=PaperPerception(width,height,cfg_cam);axis=ThrustAxisFilter(cfg_cam)
    activity=StreamingActivity(width,height);motion=prior.robust.Predictor();model=Paired()
    index=0;latest=None;last_id=None;last_label_t=None;count=0;events_total=0;beg=time.perf_counter();timings=[]
    observed=0;periodic=0;body_valid=0;fresh=0;draw=[];trace=[];last_progress=beg
    with (dest/'forecasts.jsonl').open('x',encoding='utf-8') as f:
        for t,e in raw_chunks(reader,20.,duration,20000):
            started=time.perf_counter();new=None
            while index<len(labels) and labels[index]['t']<=t+1e-9:
                r=labels[index];index+=1
                if r['t']>=20-1e-9:new=r
                latest=r
            bbox=None if latest is None or t-latest['t']>.05+1e-9 else latest['bbox']
            if new is not None:
                assert new['t']<=t+1e-9
                changed=last_id is not None and new['identity'] is not None and new['identity']!=last_id
                if changed or (last_label_t is not None and new['t']-last_label_t>.5):
                    geometry=PaperPerception(width,height,cfg_cam);axis=ThrustAxisFilter(cfg_cam);model=Paired();motion=prior.robust.Predictor()
                last_id=new['identity'];last_label_t=new['t']
            center=None if new is None else ((np.array(new['bbox'][:2])+new['bbox'][2:])/2).tolist()
            obs=dict(t=t,detected=center is not None,center=center,confidence=1. if center is not None else 0.)
            q,_=motion.step(t,center)
            velocity=np.array(q['coefficients']['cv']['velocity']) if q['valid'] else np.zeros(2)
            roi_events=e[:0] if bbox is None else e[(e['x']>=max(0,bbox[0]))&(e['x']<min(width,bbox[2]))&
                (e['y']>=max(0,bbox[1]))&(e['y']<min(height,bbox[3]))]
            rotor,_=geometry.detect(roi_events,t);n=axis.step(t,rotor.get('rotors',[]),0.,True)
            age=None if axis.last_measurement_t is None else t-axis.last_measurement_t
            q['feature_diagnostic']=dict(axis_age_s=age,models={'adaptive_axis_only':{'normal':None if n is None else n.tolist()}})
            density=np.bincount(e['y'].astype(int)*width+e['x'],minlength=width*height).reshape(height,width).astype(np.float32)
            body=dict(valid=False,reason='no_past_roi') if bbox is None else body_measurement(density,bbox)
            a=activity.step(e,bbox,t,velocity)
            v=model.step(q,obs,body,a,axis.quality)
            elapsed=(time.perf_counter()-started)*1000;timings.append(elapsed)
            v['input_audit']=dict(past_label_t=None if bbox is None else latest['t'],current_label_t=None if new is None else new['t'],
                bbox=bbox,rotors=len(rotor.get('rotors',[])),body_valid=body.get('valid',False),
                periodic_candidates=sum(c['qualified'] for c in a.get('spectral_candidates',[])),event_count=len(e),processing_ms=elapsed)
            f.write(json.dumps(v,allow_nan=False)+'\n');count+=1;events_total+=len(e);observed+=int(new is not None)
            periodic+=bool(a.get('frequency_squared'));body_valid+=bool(body.get('valid'));fresh+=v['diagnostics']['fresh']
            if count<=150:trace.append(v)
            if count%50==0:
                image=cv2.cvtColor(np.uint8(np.clip(np.log1p(density)/4,0,1)*255),cv2.COLOR_GRAY2BGR)
                if bbox is not None:cv2.rectangle(image,tuple(np.maximum([0,0],np.int32(bbox[:2]))),tuple(np.minimum([width-1,height-1],np.int32(bbox[2:]))),(0,200,0),1)
                for name,color in [('cv',(255,130,50)),('event',(0,220,255)),('constant',(220,50,220))]:
                    if v['valid']:
                        point=prior.evaluate(v['coefficients'][name],.4)
                        if np.isfinite(point).all() and max(map(abs,point))<100000:cv2.circle(image,tuple(np.int32(point)),5,color,-1)
                cv2.putText(image,f'FRED {seq} t={t:.2f} | past-box diagnostic | physics={int(v["diagnostics"]["fresh"])}',(12,25),0,.55,(230,230,230),1)
                draw.append(cv2.resize(image,(640,360)))
            if time.perf_counter()-last_progress>15:
                print(json.dumps(dict(seq=seq,t=t,ticks=count,current_physics=fresh,phase='RAW + past-box forecast diagnostic')),flush=True);last_progress=time.perf_counter()
    if prefix_duration is not None:
        original=OUT/'fred'/f'sequence{seq}'/'forecasts.jsonl'
        reference=[json.loads(s) for s in original.open(encoding='utf-8')][:count]
        rr=[json.loads(s) for s in (dest/'forecasts.jsonl').open(encoding='utf-8')]
        for old,new in zip(reference,rr):
            old['input_audit'].pop('processing_ms');new['input_audit'].pop('processing_ms');assert old==new,(seq,new['t'],'RAW-prefix')
    if draw:
        cv2.imwrite(str(dest/'fixed_contact_sheet.png'),np.vstack([np.hstack(draw[i:i+3]) for i in range(0,min(9,len(draw)),3)]))
    result=dict(sequence=seq,role=cfg['role'],ticks=count,events=events_total,position_updates=observed,
        qualified_periodic_ticks=periodic,body_axis_ticks=body_valid,current_physics_ticks=fresh,
        module_median_ms=float(np.median(timings)),module_p95_ms=float(np.percentile(timings,95)),wall_s=time.perf_counter()-beg,
        actual_RAW=True,past_provided_boxes_in_inference=True,RAW_only_deployment=False,prefix_exact=prefix_duration is not None,
        raw_sha256=cfg['raw_sha256'],source_hashes_unchanged=p['source_hashes']==sources())
    save(dest/'completion.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','self','fred']);ap.add_argument('--case');ap.add_argument('--sequence',type=int);ap.add_argument('--prefix',type=float)
    a=ap.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='self':self_run(a.case)
    else:fred_run(a.sequence,a.prefix)
