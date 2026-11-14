"""Frozen RAW geometry ablation + matched cached body/axis forecasting controls."""
from pathlib import Path
import argparse, collections, datetime, hashlib, json, sys, time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage110'),str(ROOT/'research/stage109'),
    str(ROOT/'research/stage104'),str(ROOT/'research/stage107')]
from merge110 import MergeRecovery,PARAMS
from audit109 import save,sha,rows
from windows21 import Observer
from rigid104 import PlanarCenter,PARAMS as CENTER_PARAMS
from center107 import CenterBank,NAMES
from event_drone_forecast.raw_io import open_raw,raw_chunks

OUT=ROOT.parent/'output/stage110_merge_safe_geometry_20261022'
CASES=dict(
    clear6=dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=5.,reference='output/stage104_joint_center_20261022/raw/clear6/observations_forecasts.jsonl'),
    continuous6=dict(raw='E:/dataset/train/6/media/event.raw',start=110.,duration=9.5,reference='output/stage106_continuous_screen_20261022/raw/observations_forecasts.jsonl'),
    side6=dict(raw='E:/dataset/train/6/media/event.raw',start=108.,duration=8.,reference='output/stage104_joint_center_20261022/raw/side6/observations_forecasts.jsonl'),
    negative5=dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=.6,reference='output/stage107_center_bias_filter_20261022/raw/negative5/observations_forecasts.jsonl'))

def sources():
    paths=[Path(__file__),ROOT/'research/stage110/merge110.py',ROOT/'research/stage104/rigid104.py',
        ROOT/'research/stage58/geometry58.py',ROOT/'research/stage107/center107.py',
        ROOT/'research/stage61/preline61.py',ROOT/'research/stage9/recovery.py',
        ROOT/'src/event_drone_forecast/rotor_sensing.py',ROOT/'src/event_drone_forecast/perception.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def freeze():
    assert not OUT.exists()
    save(OUT/'protocol110.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources=sources(),parameters=PARAMS,center_parameters=CENTER_PARAMS,cases=CASES,
        reference_sha256={k:sha(ROOT.parent/v['reference']) for k,v in CASES.items()},
        design='Three separate causal geometry streams: unchanged closing(line-first control), globally no closing, conditional merge preservation. Same fixed104 conic fitting/5-frame lock and107 body-only/EMA/fixed-bias/view-noise bank for each. No parameter search.',
        reason='109 C112.02 evidence: raw/open two left components become one after3x3 closing. Constants inherited from existing fill/quality/scale gates except explicitly hypothesized .15 component fraction/.7 coverage, fixed before these branch outputs.',
        scope='New RAW geometry extraction; NN body boxes and body-axis from contemporaneous old end-to-end RAW outputs replayed for matched downstream comparison. Not a new complete RAW-only end-to-end deployment.',
        tests='True truncated RAW prefix replay; data selection uses previously examined development videos. Clear/side existing sparse BOX targets only scored after forecasts; no COM/RPM/pose truth and continuous/side overlap, not independent recordings.',
        innovation='Generic component/morphology repair is not claimed new. This iteration tests a necessary measurement improvement before deriving event-rotor-specific pose/translation separation.',
        default_promoted=False))

def verify():
    p=json.loads((OUT/'protocol110.json').read_text('utf-8'));assert p['sources']==sources()
    return p

def run(case,prefix=False):
    p=verify();cfg=CASES[case];folder=OUT/(case+('_prefix' if prefix else ''));folder.mkdir(exist_ok=False)
    assert sha(ROOT.parent/cfg['reference'])==p['reference_sha256'][case]
    refs={round(r['t'],8):r for r in rows(ROOT.parent/cfg['reference'])}
    reader=open_raw(cfg['raw']);h,w=reader.get_size();models={}
    for mode in PARAMS['modes']:
        observer=Observer(w,h,'native20');observer.geometry=MergeRecovery(w,h,mode)
        models[mode]=(observer,PlanarCenter(),CenterBank())
    begin=max(0.,cfg['start']-.1);duration=min(cfg['duration'],1.2) if prefix else cfg['duration']
    count=0;cost=[]
    with (folder/'records.jsonl').open('x',encoding='utf-8') as stream:
        for t,e in raw_chunks(reader,begin,duration+cfg['start']-begin,20000):
            geometry={}
            for mode,(observer,_,_) in models.items():
                geometry[mode]=observer.step(e,t)['observation'] if t-begin>=.1-1e-8 else dict(t=t,detected=False,rotors=[])
            if t<=cfg['start']+1e-8:continue
            r=refs[round(t,8)];outputs={};tick=time.perf_counter()
            for mode,(observer,center,bank) in models.items():
                obs,d=center.step(t,r['body_observation'],geometry[mode].get('rotors',[]),r['body_axis'])
                q=bank.step(t,r['body_observation'],d,r['body_axis'])
                outputs[mode]=dict(geometry=geometry[mode],center=d,forecast=q,merge=observer.geometry.merge_audit,
                    candidate_count=len(observer.geometry.candidates))
            # Same body measurements and same KF: identical body-only control.
            body=[o['forecast']['coefficients'].get(NAMES[0]) for o in outputs.values()];assert body[0]==body[1]==body[2]
            row=dict(t=t,body=r['body_observation'],axis=r['body_axis'],valid=r['frontends']['delayed_feedback']['valid'],outputs=outputs)
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
            cost.append((time.perf_counter()-tick)*1000);count+=1
            if count%100==0:print(f'{case} {count} frames',flush=True)
    assert count==round(duration*50)
    save(folder/'checks110.json',dict(frames=count,raw_geometry=True,cached_current_body=True,labels_read=False,
        matched_body_KF_exact=True,sources_unchanged=verify()['sources']==sources(),downstream_three_banks_median_ms=float(np.median(cost))))
    if prefix:
        full=rows(OUT/case/'records.jsonl')[:count];truncated=rows(folder/'records.jsonl')
        def stable(x):
            if isinstance(x,dict):return {k:stable(v) for k,v in x.items() if k not in ['elapsed_ms','processing_ms']}
            if isinstance(x,list):return [stable(v) for v in x]
            return x
        assert stable(full)==stable(truncated),'Actual RAW prefix changed'
        save(folder/'prefix110.json',dict(actual_truncated_RAW=True,frames=count,exact_except_timing=True))
    print(json.dumps(dict(case=case,prefix=prefix,frames=count)),flush=True)

def summarize():
    verify();summary=[];details=[]
    for case in CASES:
        rr=rows(OUT/case/'records.jsonl')
        for mode in PARAMS['modes']:
            records=[r['outputs'][mode] for r in rr];streak=best=0
            for o in records:
                streak=streak+1 if o['center']['measurement']['valid'] else 0;best=max(best,streak)
            four=sum(len(o['geometry'].get('rotors',[]))==4 for o in records)
            summary.append(dict(case=case,mode=mode,frames=len(rr),four_candidates=four,
                qualified_conic=sum(o['center']['measurement']['valid'] for o in records),
                current_center=sum(o['center']['current'] for o in records),
                initializations=sum(o['center']['initialized_now'] for o in records),max_qualified_streak=best,
                split_parents=sum(o['merge'].get('split_parents',0) for o in records),
                reasons=dict(collections.Counter(o['center']['measurement']['reason'] for o in records))))
    save(OUT/'availability110.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','prefix','summarize']);p.add_argument('--case',choices=CASES)
    a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='summarize':summarize()
    else:run(a.case,a.action=='prefix')
