"""Actual RAW integration and full-state regression to frozen62 causal replay."""
import argparse, datetime, sys
import cv2
from pathlib import Path
import study62 as current
ROOT=current.ROOT;sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
import run_axis_forecast as runner
OUT=current.OUT/'raw'
REFERENCE=current.REFERENCE
load=current.load;save=current.save;sha=current.sha;rows=current.rows
configs=load(REFERENCE/'protocol.json')['cases']
CASES={c:dict(configs['off_'+c],axis_mode='axis_learn') for c in current.CASES}
CASES.update(pilot_shared_clear6=dict(CASES['clear6'],duration=.8,axis_mode='shared'),
             prefix_side6=dict(CASES['side6'],duration=4.5),
             prefix_train3=dict(CASES['train3'],duration=1.2))
SCHEDULE=['pilot_shared_clear6','clear6','side6','train3','negative5','prefix_side6','prefix_train3']


def sources():
    result=current.sources()
    for p in [Path(__file__),ROOT/'core/run_axis_forecast.py']:
        result[p.relative_to(ROOT).as_posix()]=sha(p)
    return result


def freeze():
    current.verify();assert not OUT.exists()
    assert load(current.OUT/'score_checks.json')['baseline_rows_exact61']==180
    namespace=runner.build('axis_learn',32)
    assert len(namespace['MODELS'])==15;namespace.clear()
    raws={}
    for c in CASES.values():
        p=Path(c['raw']);s=p.stat();raws[str(p)]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         source_hashes=sources(),derived_parent_sha256=current.hashlib.sha256(runner.pipeline_source().encode()).hexdigest(),
         cases=CASES,raw_metadata=raws,schedule=SCHEDULE,
         cached_forecast_hashes={c:sha(current.OUT/c/'forecasts.jsonl') for c in current.CASES},
         design='RAW-only62 axis_learn on original14s plus shared0.8s control and two actual RAW prefixes. '
                'Same61 preline/rotor sensing/body detection. Every full15/lifecycle must equal '
                'already frozen saved-input replay; no truth/cache input in deployed inference. '
                'Preview purple is changed axis-only branch; original axis+frequency expert retained.',
         accuracy_scope='Actual integration validation, not a new independent dataset. '
                        'Replay sparse scores are reusable only after exact RAW regression.'))
    (OUT/'derived_parent62.py').write_text(runner.pipeline_source(),encoding='utf-8')
    print('62 actual RAW protocol frozen',flush=True)


def verify():
    current.verify();p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    for name,s in p['raw_metadata'].items():
        v=Path(name).stat();assert s==dict(bytes=v.st_size,mtime_ns=v.st_mtime_ns)
    for c,s in p['cached_forecast_hashes'].items():assert s==sha(current.OUT/c/'forecasts.jsonl')
    return p


def run(case):
    verify();c=CASES[case]
    runner.run(c['raw'],OUT/case,c['start'],c['duration'],32,not case.startswith('prefix_'),True,c['axis_mode'])


def audit(case):
    verify();c=CASES[case];ref=c['reference'];rr=rows(OUT/case/'observations_forecasts.jsonl')
    old=rows(REFERENCE/f'off_{ref}/observations_forecasts.jsonl');ticks=rows(OUT/case/'rotor_ticks.jsonl')
    oldticks=rows(REFERENCE/f'off_{ref}/rotor_ticks.jsonl');cached=rows(current.OUT/ref/'forecasts.jsonl')
    assert len(rr)==round(c['duration']*50) and len(ticks)==2*len(rr)
    try:
        for a,b,d in zip(rr,old,cached):
            for key in ['t','event_count','filtered_event_count','geometry20','body_observation','body_boxes','observation','sensing','slot_audit','frame_sha256','body_hint_sent','roi_diagnostic','event_fields_sha256','event_packet_nbytes']:
                assert a[key]==b[key],(a['t'],key)
            m=c['axis_mode'];q=a['frontends']['delayed_feedback']
            assert q==d['frontends'][m],(a['t'],'complete15 vs causal replay')
            assert a['lifecycle']['delayed_feedback']==d['lifecycle'][m],(a['t'],'lifecycle')
        for a,b in zip(ticks,oldticks):
            for key in ['t','event_count','geometry_event_count','observation','sensing','slot_audit','roi_diagnostic']:
                assert a[key]==b[key],(a['t'],'native10 '+key)
    except AssertionError as error:
        save(OUT/case/'regression_failure62.json',dict(error=str(error)));raise
    completion=load(OUT/case/'completion.json');native=load(OUT/case/'rotor_completion.json')
    assert completion['source_hashes_unchanged'] and native['source_hashes_unchanged']
    assert not completion['labels_read'] and not native['labels_read']
    video=0
    if (OUT/case/'forecasts.mp4').exists():
        cap=cv2.VideoCapture(str(OUT/case/'forecasts.mp4'))
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape[:2]==(720,1280);video+=1
        cap.release();assert video==len(rr)
    summary=dict(case=case,parent=len(rr),native10=len(ticks),video_frames_decoded=video,
        all_current_inputs_exact61=True,complete15_and_lifecycle_exact62_replay=True,
        labels_read=False,processing_median_ms=completion['processing_median_ms'])
    save(OUT/case/'regression62.json',summary);print(summary,flush=True)


def prefix():
    verify();result=[]
    for name,ref in [('prefix_side6','side6'),('prefix_train3','train3')]:
        a=rows(OUT/name/'observations_forecasts.jsonl');b=rows(OUT/ref/'observations_forecasts.jsonl')
        for x,y in zip(a,b):
            for key in ['t','observation','sensing','slot_audit','frontends','lifecycle','body_hint_sent','roi_diagnostic','event_fields_sha256','event_packet_nbytes']:
                assert x[key]==y[key],(name,x['t'],key)
        a10=rows(OUT/name/'rotor_ticks.jsonl');b10=rows(OUT/ref/'rotor_ticks.jsonl')
        for x,y in zip(a10,b10):
            for key in ['t','observation','sensing','slot_audit','roi_diagnostic']:assert x[key]==y[key]
        result.append(dict(case=ref,parent=len(a),native10=len(a10)))
    totals=dict(parent=0,native10=0,video_frames_decoded=0)
    for case in SCHEDULE:
        d=load(OUT/case/'regression62.json')
        for key in totals:totals[key]+=d[key]
    save(OUT/'raw_prefix_checks.json',dict(table=result,totals=totals,actual_RAW_prefix=True,
         full15_and_lifecycle_exact=True,all_native_current_exact61=True,
         forecast_scores_reusable_from62_replay=True))
    print('62 actual RAW prefixes complete: '+str(totals),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','audit','prefix']);p.add_argument('--case',choices=SCHEDULE);a=p.parse_args()
    if a.action in ['run','audit']:globals()[a.action](a.case)
    else:globals()[a.action]()
