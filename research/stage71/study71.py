"""Freeze, run and independently check actual RAW event-force integration."""
from pathlib import Path
import argparse,datetime,importlib.util,json,subprocess,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research/stage70'),str(ROOT/'research/stage69'),str(ROOT/'research/stage68'),str(ROOT/'research/stage64'),str(ROOT/'research/stage22')]
import run_event_force_forecast as entry
import study70 as cache
base=cache.base;activity=cache.previous.activity;body=cache.previous.body
load=base.load;save=base.save;sha=base.sha;rows=base.rows
OUT=ROOT.parent/'output/stage71_RAW_event_force_20261021'
CASES=dict(cache.CASES)
CASES.update(prefix_side6=dict(CASES['side6'],duration=4.5),prefix_train3=dict(CASES['train3'],duration=1.2))

def sources():
    ns=entry.build();hashes=ns['fingerprints']();ns.clear()
    hashes['research/stage71/study71.py']=sha(__file__)
    return hashes
def freeze():
    cache.verify();assert not (OUT/'protocol.json').exists()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,
        previous68_protocol_sha=sha(activity.OUT/'protocol.json'),previous69_protocol_sha=sha(cache.previous.OUT/'protocol.json'),
        previous70_protocol_sha=sha(cache.OUT/'protocol.json'),
        design='Actual RAW independent entry computes current NN, geometry, activity, '
               'motion-compensated body direction, force/RLS and guard inline. '
               'No prior caches or labels can be read by inference. Main14s plus '
               'actual4.5s/1.2s truncation prefixes. Compare full26 predictions '
               'against causal cache ablation only in separate post-run checks.',
        labels_read_by_inference=False,default_promoted=False))
    print('71 actual RAW protocol frozen',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources();return p
def run(case):
    verify();cfg=CASES[case];folder=OUT/case
    args=[sys.executable,str(ROOT/'run_event_force_forecast.py'),'--raw',cfg['raw'],'--output',str(folder),
        '--start',str(cfg['start']),'--duration',str(cfg['duration']),'--verify-frames']
    if not case.startswith('prefix') and case!='negative5':args.append('--preview')
    with (OUT/(case+'_parent.log')).open('x',encoding='utf-8') as f:
        result=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert result.returncode==0,(case,'inference failed',result.returncode)
    check(case);print(case,'actual RAW full26 checked',flush=True)
def clean_body(m):return {k:v for k,v in m.items() if k!='undirected_disagreement_with_stored_rotor_axis_deg'}
def check(case):
    verify();cfg=CASES[case];reference=case.removeprefix('prefix_')
    rr=rows(OUT/case/'observations_forecasts.jsonl');n=round(cfg['duration']*50);assert len(rr)==n
    original=rows(base.REFERENCE/f'off_{reference}/observations_forecasts.jsonl')[:n]
    force=rows(cache.OUT/reference/'forecasts.jsonl')[:n];aa=rows(activity.OUT/reference/'activity.jsonl')[:n];bb=rows(body.OUT/reference/'observations.jsonl')[:n]
    try:
        for r,old,f,a,b in zip(rr,original,force,aa,bb):
            for k in ['t','geometry20','body_observation','observation','sensing','slot_audit','body_boxes','frame_sha256','event_fields_sha256']:
                assert r[k]==old[k],(r['t'],k)
            assert r['event_activity']==a['activity'],(r['t'],'activity68')
            assert r['event_measurement_audit']==a['velocity_audit'],(r['t'],'velocity68')
            assert r['body_axis']==clean_body(b['methods']['aligned_total']),(r['t'],'body64')
            assert r['frontends']['delayed_feedback']==f['frontends']['event_force'],(r['t'],'full26_forecast_and_learning')
        if case.startswith('prefix_'):
            full=rows(OUT/reference/'observations_forecasts.jsonl')[:n]
            for r,f in zip(rr,full):
                for k in ['t','event_activity','body_axis','event_measurement_audit','frontends','lifecycle','event_fields_sha256']:
                    assert r[k]==f[k],(r['t'],'actualRAWprefix',k)
    except Exception as error:
        save(OUT/case/'regression_failure71.json',dict(error=str(error),t=r.get('t'),actual=r,expected_original=old,expected_forecast=f));raise
    frames=0;video=OUT/case/'forecasts.mp4'
    if video.is_file():
        cap=cv2.VideoCapture(str(video))
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(720,1280,3);frames+=1
        cap.release();assert frames==n
    completion=load(OUT/case/'completion.json');assert completion['source_hashes_unchanged'] and not completion['labels_read']
    save(OUT/case/'regression71.json',dict(parent=n,native10=2*n,original_current_exact61=True,
        activity_and_past_velocity_exact68=True,body_measurement_exact64=True,
        full26_forecasts_and_learning_exact70=True,actual_RAW_prefix=case.startswith('prefix_'),video_frames_decoded=frames))
def summarize():
    verify();table=[]
    for case in CASES:table.append(dict(case=case,**load(OUT/case/'regression71.json'),completion=load(OUT/case/'completion.json')))
    save(OUT/'checks71.json',dict(table=table,parent=sum(r['parent'] for r in table),native10=sum(r['native10'] for r in table),
        video_frames_decoded=sum(r['video_frames_decoded'] for r in table),actual_RAW_only=True,default_promoted=False,
        scope='Main14s/700 current ticks plus285 repeated prefix ticks; not985 independent observations. Pilot separately recorded.'))
    print('71 all actual RAW regressions/prefixes complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','check','summarize']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action in ['run','check']:globals()[a.action](a.case)
    else:globals()[a.action]()
