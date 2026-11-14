"""Full RAW CV/physical IMM, independently verified against frozen74 replay."""
from pathlib import Path
import argparse,datetime,subprocess,sys
import cv2
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage74')]
import run_hybrid_forecast as entry
import study74 as cache
base=cache.base;load=base.load;save=base.save;sha=base.sha;rows=base.rows
OUT=ROOT.parent/'output/stage75_RAW_parallel_forecast_20261021';CASES={c:dict(cfg) for c,cfg in cache.CASES.items()}
CASES.update(prefix_side6=dict(CASES['side6'],duration=4.5),prefix_train3=dict(CASES['train3'],duration=1.2))
def sources():
    ns=entry.build();h=ns['fingerprints']();ns.clear();h['research/stage75/study75.py']=sha(__file__);return h
def freeze():
    cache.verify();assert not OUT.exists()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,
        upstream=cache.snapshot,design='Actual RAW-only current NN + activity + force + '
        'full author IMM interaction. Original33 outputs unchanged. All39 '
        'forecasts/learning/mode probabilities checked against74 only after '
        'inference exits. Main23s and two actual RAW truncation prefixes.',
        labels_read_by_inference=False,default_promoted=False,
        scope='Engineering integration, not a realtime/high-accuracy guarantee. '
              'Cache74 already found side/train3 degradation vs strong history; '
              'negative results remain part of the deliverable.'))
    print('75 RAW dual-branch protocol frozen',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources();return p
def run(case):
    verify();cfg=CASES[case];args=[sys.executable,str(ROOT/'core/run_hybrid_forecast.py'),'--raw',cfg['raw'],'--output',str(OUT/case),
        '--start',str(cfg['start']),'--duration',str(cfg['duration']),'--verify-frames']
    if not case.startswith('prefix') and case!='negative5':args.append('--preview')
    with (OUT/(case+'_parent.log')).open('x',encoding='utf-8') as f:
        result=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert result.returncode==0,(case,result.returncode);check(case);print(case,'75 full39 actual RAW exact',flush=True)
def check(case):
    verify();ref=case.removeprefix('prefix_');rr=rows(OUT/case/'observations_forecasts.jsonl');n=round(CASES[case]['duration']*50);assert len(rr)==n
    original=rows(cache.previous.OUT/ref/'observations_forecasts.jsonl')[:n];forecast=rows(cache.OUT/ref/'forecasts.jsonl')[:n]
    try:
        for r,o,e in zip(rr,original,forecast):
            for k in ['t','event_activity','body_axis','event_measurement_audit','geometry20','body_observation','observation','sensing','slot_audit','body_boxes','frame_sha256','event_fields_sha256']:
                assert r[k]==o[k],(r['t'],k)
            assert r['frontends']['delayed_feedback']==e['frontends']['event_force'],(r['t'],'full39_and_mode_state')
        if case.startswith('prefix'):
            full=rows(OUT/ref/'observations_forecasts.jsonl')[:n]
            for r,o in zip(rr,full):
                for k in ['t','event_activity','body_axis','event_measurement_audit','frontends','lifecycle','event_fields_sha256']:assert r[k]==o[k],(r['t'],'actualprefix',k)
    except Exception as error:
        save(OUT/case/'regression_failure75.json',dict(error=str(error),actual=r,expected_original=o,expected_forecast=e));raise
    frames=0;video=OUT/case/'forecasts.mp4'
    if video.is_file():
        cap=cv2.VideoCapture(str(video))
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(720,1280,3);frames+=1
        cap.release();assert frames==n
    c=load(OUT/case/'completion.json');assert c['source_hashes_unchanged'] and not c['labels_read']
    save(OUT/case/'regression75.json',dict(parent=n,native10=2*n,all_current_exact73=True,
        full39_forecasts_parameters_covariances_mode_probabilities_exact74=True,
        actual_RAW_prefix=case.startswith('prefix'),video_frames_decoded=frames))
def summarize():
    verify();table=[dict(case=c,**load(OUT/c/'regression75.json'),completion=load(OUT/c/'completion.json')) for c in CASES]
    save(OUT/'checks75.json',dict(table=table,parent=sum(r['parent'] for r in table),native10=sum(r['native10'] for r in table),
        video_frames_decoded=sum(r['video_frames_decoded'] for r in table),default_promoted=False,actual_RAW_only=True))
    print('75 all RAW checks/video/prefixes complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','check','summarize']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action in ['run','check']:globals()[a.action](a.case)
    else:globals()[a.action]()
