"""Frozen motion-compensated body-axis measurement study, no future labels."""
from pathlib import Path
from collections import Counter
import argparse,datetime,sys,time
import cv2,numpy as np
from compensation64 import *
sys.path.insert(0,str(ROOT/'research/stage63'));import study63 as prior
from event_drone_forecast.raw_io import open_raw,raw_chunks
OUT=ROOT.parent/'output/stage64_motion_compensated_axis_20261021'
CASES=dict(prior.CASES);CASES.update(prefix_side6=dict(CASES['side6'],duration=4.5),prefix_train3=dict(CASES['train3'],duration=1.2))
MAIN=list(prior.CASES);load=prior.load;save=prior.save;sha=prior.sha;rows=prior.rows


def sources():
    files=[Path(__file__),Path(__file__).with_name('compensation64.py'),ROOT/'research/stage21/windows21.py',
           ROOT/'research/stage6/silhouette.py',ROOT/'research/stage8/polarity.py',ROOT/'src/event_drone_forecast/raw_io.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files}


def freeze():
    prior.verify();assert not OUT.exists()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        parameters=PARAMS,cases=CASES,reference63_sha256=sha(prior.OUT/'protocol.json'),
        body_input_hashes={c:sha(prior.REF/f'off_{c}/observations_forecasts.jsonl') for c in MAIN},
        observations63_hashes={c:sha(prior.OUT/c/'observations.jsonl') for c in MAIN},
        raw_metadata=load(prior.OUT/'protocol.json')['raw_metadata'],
        design='Same current NN bbox/body-quality gate and unchanged6 PCA/LSD, static control '
               'must match63 complete measurements. Aligned arm adds exact21 event warp and6 '
               'density translation using only previously committed body-supported CV velocity. '
               'Two actual RAW truncation prefixes; no forecast scores or adaptation.',
        prior_art=['https://openaccess.thecvf.com/content_cvpr_2018/html/Gallego_A_Unifying_Contrast_CVPR_2018_paper.html',
                   'https://github.com/uzh-rpg/event-based_vision_resources'],
        reuse='Local6 body_measurement/image translation and21 warp; standard compensation, '
              'not a newly implemented contrast-maximization optimizer. No new packages.',
        scope='Development recordings, cached same-time NN boxes and strictly earlier CV '
              'states. Observation-only, not a standalone RAW joint inference entry.'))
    print('64 compensation protocol frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    assert p['reference63_sha256']==sha(prior.OUT/'protocol.json')
    for c,v in p['body_input_hashes'].items():assert v==sha(prior.REF/f'off_{c}/observations_forecasts.jsonl')
    for c,v in p['observations63_hashes'].items():assert v==sha(prior.OUT/c/'observations.jsonl')
    for n,v in p['raw_metadata'].items():
        s=Path(n).stat();assert v==dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    return p


def synthetic_run():
    verify();assert not (OUT/'synthetic64.json').exists();save(OUT/'synthetic64.json',synthetic())
    print('64 known-orientation translation and wrong-sign controls complete',flush=True)


def run(case):
    verify();sys.addaudithook(prior.previous.guard);cfg=CASES[case];ref=cfg['reference']
    folder=OUT/case;folder.mkdir(exist_ok=False)
    rr=rows(prior.REF/f'off_{ref}/observations_forecasts.jsonl');old63=rows(prior.OUT/ref/'observations.jsonl')
    old62=rows(prior.previous.OUT/ref/'forecasts.jsonl')
    reader=open_raw(Path(cfg['raw']));h,w=reader.get_size();memory=Supports(w,h);velocity=PastVelocity();result=[]
    stats={m:Counter() for m in memory.density};reason={m:Counter() for m in memory.density};timing={m:[] for m in memory.density}
    count=valid_velocity=0;fixed={round((cfg['start']+cfg['duration']*f)*50)/50 for f in [.25,.5,.75]}
    for i,(t,e) in enumerate(raw_chunks(reader,cfg['start'],cfg['duration'],10000)):
        begin=t-.01;v,va=velocity.before(begin);assert va['source_t'] is None or va['source_t']<=begin+1e-8
        packet=memory.step(e,t,v);count+=1;valid_velocity+=va['valid']
        if (i+1)%2:continue
        j=len(result);r=rr[j];assert abs(t-r['t'])<1e-8;body=r['body_observation'];methods={}
        a=old62[j]['frontends']['axis_learn']['feature_diagnostic']['models']['adaptive_axis_only']
        for name,density in memory.density.items():
            started=time.perf_counter()
            if body.get('detected') and body.get('confidence',0)>=PARAMS['body_confidence_min']:
                m=body_measurement(density,body['bbox']);stats[name]['body_supported']+=1
            else:m=dict(valid=False,reason='no_current_NN_identity')
            timing[name].append((time.perf_counter()-started)*1000);reason[name][m['reason']]+=1
            if m['valid']:
                stats[name]['valid']+=1;stats[name]['while62_axis_inactive']+=not a['active']
                if a['normal'] is not None:
                    n=np.asarray(a['normal'][:2],float);axis=np.asarray(m['normal_axis_xy'],float)
                    if np.linalg.norm(n)>.05:m['undirected_disagreement_with_stored_rotor_axis_deg']=float(np.degrees(np.arccos(np.clip(abs(n@axis)/np.linalg.norm(n),0,1))))
            if name.startswith('static_'):
                key='total_decay' if name.endswith('_total') else 'bipolar_decay'
                if m!=old63[j]['methods'][key]:
                    save(folder/'regression_failure64.json',dict(t=t,method=name,actual=m,expected=old63[j]['methods'][key]));raise AssertionError('Static control differs63')
            methods[name]=m
            if round(t,8) in fixed:
                im=np.uint8(np.clip(np.log1p(density)/4,0,1)*255);im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
                if body.get('detected'):
                    b=np.rint(body['bbox']).astype(int);lo=np.maximum(b[:2]-30,[0,0]);hi=np.minimum(b[2:]+30,[w,h]);cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,180,255),1)
                    if 'angle_rad' in m:
                        c=(b[:2]+b[2:])/2;u=np.array([np.cos(m['angle_rad']),np.sin(m['angle_rad'])]);size=max(b[2:]-b[:2])*.4
                        cv2.line(im,tuple(np.rint(c-u*size).astype(int)),tuple(np.rint(c+u*size).astype(int)),(60,220,80) if m['valid'] else (60,60,230),2)
                    im=im[lo[1]:hi[1],lo[0]:hi[0]]
                cv2.imencode('.png',im)[1].tofile(str(folder/f'{name}_{t:.2f}.png'))
        result.append(dict(t=t,methods=methods,velocity=v.tolist(),velocity_audit=va,packet=packet,
                           axis62_active=a['active'],forecast_valid=old62[j]['frontends']['axis_learn']['valid']))
        velocity.commit(r)  # Current20ms position enters only after current geometry is complete.
    assert count==round(cfg['duration']*100) and len(result)==count//2
    with (folder/'observations.jsonl').open('x',encoding='utf-8') as f:
        for r in result:f.write(prior.json.dumps(r,allow_nan=False)+'\n')
    summary=dict(case=case,parent=len(result),native10=count,velocity_supported_native10=valid_velocity,
        static_complete_measurements_exact63=True,velocity_only_known_before_packet_start=True,
        methods={m:dict(stats[m],reasons=dict(reason[m]),measurement_median_ms=float(np.median(timing[m]))) for m in stats},
        future_predictions_executed=False,labels_read=False)
    save(folder/'completion.json',summary);print(summary,flush=True)


def prefix():
    verify();result=[]
    for alias,ref in [('prefix_side6','side6'),('prefix_train3','train3')]:
        a=rows(OUT/alias/'observations.jsonl');b=rows(OUT/ref/'observations.jsonl')[:len(a)];assert a==b
        result.append(dict(case=ref,parent=len(a),native10=2*len(a),all_measurements_velocity_and_packet_audit_exact=True))
    totals=dict(parent=0,native10=0)
    for c in CASES:
        d=load(OUT/c/'completion.json')
        for k in totals:totals[k]+=d[k]
    save(OUT/'raw_prefix_checks.json',dict(table=result,totals=totals,actual_RAW_prefix=True,
        scope='Observation and past-velocity states only; no new future prediction scores.'))
    print('64 actual RAW observation prefixes exact: '+str(totals),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','synthetic_run','run','prefix']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action=='run':run(a.case)
    else:globals()[a.action]()
