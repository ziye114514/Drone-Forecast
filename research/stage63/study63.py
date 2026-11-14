"""Whole-body side-axis observations using existing6 PCA/LSD without rotor lock."""
from pathlib import Path
from collections import Counter
import argparse, datetime, hashlib, json, os, sys, time
import cv2, numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage62'),str(ROOT/'research/stage6'),str(ROOT/'src')]
import study62 as previous
from silhouette import body_measurement, PARAMS as BODY_PARAMS
from event_drone_forecast.raw_io import open_raw,raw_chunks

OUT=ROOT.parent/'output/stage63_body_axis_observation_20261021'
REF=previous.REFERENCE
CASES={c:dict(previous.load(REF/'protocol.json')['cases']['off_'+c]) for c in previous.CASES}
METHODS=['total_decay','bipolar_decay']
PARAMS=dict(decay_s=.02,step_us=10000,publish_us=20000,body_confidence_min=.45,
    methods=METHODS,body_parameters=BODY_PARAMS,
    density='total_decay reuses6 raw event-count exponential support; bipolar_decay '
            'reuses8 min(Gaussian ON, Gaussian OFF) before identical exponential support. '
            'No G/intensity reconstruction, no frequency estimation from activity counts.',
    identity='Use only same-cutoff actual61 NN body bbox/confidence; no rotor lock, '
             'no future box, no body position/center inferred from line. Current-axis '
             'evidence only; no physical sign/RPM/pose truth or new forecast gain claim.')
load=previous.load;save=previous.save;sha=previous.sha;rows=previous.rows


def sources():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),ROOT/'research/stage6/silhouette.py',ROOT/'research/stage8/polarity.py',ROOT/'src/event_drone_forecast/raw_io.py']}


def freeze():
    assert not OUT.exists();previous.verify()
    metadata={}
    for cfg in CASES.values():
        p=Path(cfg['raw']);s=p.stat();metadata[str(p)]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_hashes=sources(),cases=CASES,parameters=PARAMS,raw_metadata=metadata,
        input_hashes={c:sha(REF/f'off_{c}/observations_forecasts.jsonl') for c in CASES},
        replay62_hashes={c:sha(previous.OUT/c/'forecasts.jsonl') for c in CASES},
        actual_reuse='Exact stage6 body_measurement with PCA/multithreshold/LSD and '
                     'unchanged thresholds. Replace obsolete four-rotor initialization '
                     'requirement with already verified current NN identity only at call site. '
                     'No four spokes/centers/rotor IDs manufactured.',
        prior_art=['https://github.com/event-driven-robotics/RT-EvLDT',
                   'https://docs.opencv.org/4.x/db/d73/classcv_1_1LineSegmentDetector.html'],
        libraries='Actual installed OpenCV5.0.0 LSD;4.x reference describes API, '
                  'not used to infer5.0 implementation details. RT-EvLDT reviewed, not run.',
        scope='Current-observation pilot on known development recordings with cached '
              'same-time NN boxes. Not a deployable RAW-only joint entry or new '
              'future/RPM/pose accuracy result. Separate held-out validation needed.'))
    print('63 body-axis observation protocol frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    for c,h in p['input_hashes'].items():assert h==sha(REF/f'off_{c}/observations_forecasts.jsonl')
    for c,h in p['replay62_hashes'].items():assert h==sha(previous.OUT/c/'forecasts.jsonl')
    for n,m in p['raw_metadata'].items():
        s=Path(n).stat();assert m==dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    return p


def run(case):
    verify();sys.addaudithook(previous.guard);cfg=CASES[case]
    target=OUT/case;target.mkdir(exist_ok=False)
    rr=rows(REF/f'off_{case}/observations_forecasts.jsonl')
    cache=rows(previous.OUT/case/'forecasts.jsonl')
    reader=open_raw(Path(cfg['raw']));h,w=reader.get_size();assert (h,w)==(720,1280)
    density={m:np.zeros((h,w),np.float32) for m in METHODS};output=[];count=0
    stats={m:Counter() for m in METHODS};reason={m:Counter() for m in METHODS}
    elapsed={m:[] for m in METHODS};decay=np.exp(-.01/PARAMS['decay_s'])
    fixed={round(cfg['start']+cfg['duration']*f,8) for f in [.25,.5,.75]}
    for i,(t,e) in enumerate(raw_chunks(reader,cfg['start'],cfg['duration'],10000)):
        count+=1;pix=e['y'].astype(np.int32)*w+e['x']
        total=np.bincount(pix,minlength=h*w).reshape(h,w).astype(np.float32)
        on=np.bincount(pix[e['p']>0],minlength=h*w).reshape(h,w).astype(np.float32)
        off=total-on
        feature=np.minimum(cv2.GaussianBlur(on,(3,3),.8),cv2.GaussianBlur(off,(3,3),.8))
        for name,value in [('total_decay',total),('bipolar_decay',feature)]:density[name]*=decay;density[name]+=value
        if (i+1)%2:continue
        j=len(output);row=rr[j];assert abs(t-row['t'])<1e-8
        body=row['body_observation'];q=cache[j]['frontends']['axis_learn'];a=q['feature_diagnostic']['models']['adaptive_axis_only']
        current_rotor_axis=cache[j]['axis_audit']['axis_learn']['current_axis_support']
        methods={}
        for name in METHODS:
            start=time.perf_counter()
            if body.get('detected') and body.get('confidence',0)>=PARAMS['body_confidence_min']:
                m=body_measurement(density[name],body['bbox']);stats[name]['body_supported']+=1
            else:m=dict(valid=False,reason='no_current_NN_identity')
            elapsed[name].append((time.perf_counter()-start)*1000)
            reason[name][m['reason']]+=1
            if m['valid']:
                stats[name]['valid']+=1
                stats[name]['without_current_rotor_axis']+=not current_rotor_axis
                stats[name]['while62_axis_inactive']+=not a['active']
                stats[name]['while_forecast_valid']+=q['valid']
                if a['normal'] is not None:
                    n=np.asarray(a['normal'][:2],float);axis=np.asarray(m['normal_axis_xy'],float)
                    if np.linalg.norm(n)>.05:
                        m['undirected_disagreement_with_stored_rotor_axis_deg']=float(np.degrees(np.arccos(np.clip(abs(n@axis)/np.linalg.norm(n),0,1))))
            methods[name]=m
            if round(t,8) in fixed:
                im=np.uint8(np.clip(np.log1p(density[name])/4,0,1)*255);im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
                if body.get('detected'):
                    b=np.rint(body['bbox']).astype(int);cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,180,255),1)
                    if 'angle_rad' in m:
                        c=(b[:2]+b[2:])/2;u=np.array([np.cos(m['angle_rad']),np.sin(m['angle_rad'])]);size=max(b[2:]-b[:2])*.4
                        color=(70,220,80) if m['valid'] else (70,70,230)
                        cv2.line(im,tuple(np.rint(c-u*size).astype(int)),tuple(np.rint(c+u*size).astype(int)),color,2)
                cv2.putText(im,f'{case} {name} t={t:.2f} | '+m['reason'],(12,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                cv2.imencode('.png',im)[1].tofile(str(target/f'{name}_{t:.2f}.png'))
        output.append(dict(t=t,methods=methods,body_detected=body.get('detected'),
            rotor_axis_current=current_rotor_axis,axis62_active=a['active'],forecast_valid=q['valid']))
    assert count==round(cfg['duration']*100) and len(output)==len(rr)
    with (target/'observations.jsonl').open('x',encoding='utf-8') as f:
        for r in output:f.write(json.dumps(r,allow_nan=False)+'\n')
    save(target/'completion.json',dict(case=case,native10=count,parent=len(output),labels_read=False,
        methods={m:dict(stats[m],reasons=dict(reason[m]),measurement_median_ms=float(np.median(elapsed[m]))) for m in METHODS},
        future_forecasts_executed=False))
    print(load(target/'completion.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action=='run':run(a.case)
    else:freeze()
