"""Actual RAW full49 and fixed additional-recording checks after structural fixes."""
from pathlib import Path
import argparse,datetime,importlib.util,json,subprocess,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage77'),str(ROOT/'research/stage75'),str(ROOT/'research/stage22')]
import run_separated_forecast as entry
import study77 as cache
previous=cache.actual;base=cache.base
load=base.load;save=base.save;sha=base.sha;rows=base.rows
OUT=ROOT.parent/'output/stage78_RAW_separated_forecast_20261021'
CASES={c:dict(cfg) for c,cfg in base.CASES.items()}
CASES.update(prefix_side6=dict(CASES['side6'],duration=4.5),prefix_train3=dict(CASES['train3'],duration=1.2),
    extra1=dict(raw='E:/dataset/train/1/event.raw',start=30.,duration=5.),
    extra2=dict(raw='E:/dataset/train/2/aligned/event.raw',start=30.,duration=5.),
    extra4=dict(raw='E:/dataset/train/4/aligned/event.raw',start=60.,duration=5.))
SCORING_CASES=[c for c in CASES if not c.startswith('prefix')]
LABELS={c:('train'+c[-1]+'.json' if c.startswith('extra') else 'train3.json' if c=='train3' else 'train6.json') for c in SCORING_CASES if c!='negative5'}
def sources():
    ns=entry.build();h=ns['fingerprints']();ns.clear();h['research/stage78/study78.py']=sha(__file__);return h
def freeze():
    cache.verify();assert not OUT.exists()
    metadata={}
    for c in CASES.values():
        p=Path(c['raw']);s=p.stat();metadata[str(p)]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    labels=base.base.BASE/'raw_only_project/evaluation_data'
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,
        label_hashes={f:sha(labels/f) for f in sorted(set(LABELS.values()))},raw_metadata=metadata,
        upstream=cache.SNAPSHOT,parameters_unchanged_from77=True,
        design='Actual RAW-only original39 + ten77 structural arms. Current '
               'observations/49 forecasts checked against77 only after inference. '
               'Two actual truncation prefixes. Fixed additional recordings: '
               'train1 30–35, train2 30–35, train4 60–65, chosen by recording '
               'duration midpoint metadata without consulting label states/errors.',
        primary_review_branch='mmae_state_fresh',
        scope='Development structures already informed by76. Additional fixed '
              'segments assess transportability without retuning, not unseen '
              'subjects or blind detector validation. All additional failures '
              'and absent-label outputs retained. Predictions before label parsing.',
        default_promoted=False))
    print('78 RAW and additional-segment protocol frozen',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    for n,m in p['raw_metadata'].items():
        s=Path(n).stat();assert m==dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    return p
def run(case):
    verify();cfg=CASES[case];args=[sys.executable,str(ROOT/'core/run_separated_forecast.py'),'--raw',cfg['raw'],'--output',str(OUT/case),
        '--start',str(cfg['start']),'--duration',str(cfg['duration']),'--verify-frames']
    if not case.startswith('prefix') and case!='negative5':args.append('--preview')
    with (OUT/(case+'_parent.log')).open('x',encoding='utf-8') as f:
        result=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert result.returncode==0,(case,result.returncode);check(case);print(case,'78 actual RAW checked',flush=True)
def check(case):
    verify();rr=rows(OUT/case/'observations_forecasts.jsonl');n=round(CASES[case]['duration']*50);assert len(rr)==n
    ref=case.removeprefix('prefix_');has_ref=ref in base.CASES
    original=rows(previous.OUT/ref/'observations_forecasts.jsonl')[:n] if has_ref else None
    expected=rows(cache.OUT/ref/'forecasts.jsonl')[:n] if has_ref else None
    try:
        for i,r in enumerate(rr):
            if has_ref:
                for k in ['t','event_activity','body_axis','event_measurement_audit','geometry20','body_observation','observation','sensing','slot_audit','body_boxes','frame_sha256','event_fields_sha256']:
                    assert r[k]==original[i][k],(r['t'],k)
                assert r['frontends']['delayed_feedback']==expected[i]['frontends']['event_force'],(r['t'],'all49')
        if case.startswith('prefix'):
            full=rows(OUT/ref/'observations_forecasts.jsonl')[:n]
            for r,o in zip(rr,full):
                for k in ['t','event_activity','body_axis','event_measurement_audit','frontends','lifecycle','event_fields_sha256']:assert r[k]==o[k],(r['t'],'actualprefix',k)
    except Exception as error:
        save(OUT/case/'regression_failure78.json',dict(error=str(error),actual=r,
            expected_original=original[i] if original else None,expected_forecast=expected[i] if expected else None));raise
    frames=0;video=OUT/case/'forecasts.mp4'
    if video.is_file():
        cap=cv2.VideoCapture(str(video))
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(720,1280,3);frames+=1
        cap.release();assert frames==n
    c=load(OUT/case/'completion.json');assert c['source_hashes_unchanged'] and not c['labels_read']
    save(OUT/case/'regression78.json',dict(parent=n,native10=2*n,current_exact75=has_ref,all49_exact77=has_ref,
        actual_RAW_prefix=case.startswith('prefix'),video_frames_decoded=frames,additional_fixed_recording=case.startswith('extra')))
def summarize():
    verify();table=[dict(case=c,**load(OUT/c/'regression78.json'),completion=load(OUT/c/'completion.json')) for c in CASES]
    save(OUT/'checks78.json',dict(table=table,parent=sum(r['parent'] for r in table),native10=sum(r['native10'] for r in table),
        video_frames_decoded=sum(r['video_frames_decoded'] for r in table),default_promoted=False,actual_RAW_only=True))
    print('78 all RAW checks/video/prefixes complete',flush=True)
def score():
    verify();assert (OUT/'checks78.json').is_file();scoring=OUT/'scoring';scoring.mkdir(exist_ok=False)
    for c in SCORING_CASES:
        folder=scoring/c;folder.mkdir()
        with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as f:
            for r in rows(OUT/c/'observations_forecasts.jsonl'):f.write(json.dumps(dict(t=r['t'],frontends=dict(event_force=r['frontends']['delayed_feedback'])),allow_nan=False)+'\n')
    save(scoring/'protocol.json',load(OUT/'protocol.json'))
    spec=importlib.util.spec_from_file_location('score78_original22',ROOT/'research/stage22/study22.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=scoring;m.INPUT=scoring;m.CASES=SCORING_CASES;m.FRONTENDS=['event_force'];m.MODELS=cache.MODELS;m.verify=verify;m.evaluate=cache.prior.evaluate
    m.label_path=lambda c:base.base.BASE/'raw_only_project/evaluation_data'/LABELS[c];m.score()
    analysis=load(scoring/'analysis.json');old=load(cache.OUT/'analysis.json');lookup={(r['case'],r['model'],r['nominal_h']):r for r in old['table']};count=0
    for r in analysis['table']:
        if r['case'] in base.CASES:assert r==lookup[r['case'],r['model'],r['nominal_h']];count+=1
        if r['case']!='negative5' and r['nominal_h']==.4 and r['model'] in ['cv','trusted_acc','dual_cv_event_increment','mmae_state_fresh','mmae_state_fresh_constant','imm_forecast_fresh']:
            print({k:r[k] for k in ['case','model','mean_fde_px','joint_success','valid_forecasts','present_targets','absent_targets','absent_forecasts_inside']},flush=True)
    save(OUT/'score_checks.json',dict(rows=len(analysis['table']),original77_rows_exact=count,additional_rules_unchanged=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','check','summarize','score']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action in ['run','check']:globals()[a.action](a.case)
    else:globals()[a.action]()
