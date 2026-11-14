"""Untuned temporal extension of the full RAW increment forecast."""
from pathlib import Path
import argparse,datetime,importlib.util,json,subprocess,sys
import cv2
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage73'),str(ROOT/'research/stage72'),str(ROOT/'research/stage71'),str(ROOT/'research/stage22')]
import run_force_innovation_forecast as entry
import study72 as cache
import study71 as previous
from online73 import controls
base=cache.base;load=base.load;save=base.save;sha=base.sha;rows=base.rows
OUT=ROOT.parent/'output/stage73_temporal_extension_20261021'
CASES={c:dict(cfg) for c,cfg in cache.CASES.items()}
CASES['clear6']['duration']=5.;CASES['side6']['duration']=8.;CASES['train3']['duration']=9.
MODELS=cache.MODELS+['increment_guarded_events','increment_guarded_constants','increment_guarded_all']
NEW_START=dict(clear6=24.,side6=113.,train3=31.)
def sources():
    ns=entry.build();h=ns['fingerprints']();ns.clear();h['research/stage73/study73.py']=sha(__file__);return h
def freeze():
    cache.verify();assert not OUT.exists()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,
        label_hashes=load(cache.OUT/'protocol.json')['label_hashes'],controls=controls(),
        new_anchor_start=NEW_START,
        design='Frozen72 increments and unchanged70 guard rules, actual RAW only. '
               'Warm from original22/108/25 starts; extend to27/116/34s. '
               'No new parameters tuned on extension labels. Score known prefix '
               'and later targets separately; supplementary targets start .4s '
               'after new-anchor cutoff so all .1/.2/.4s anchors are in extension.',
        scope='Temporal extension of development recordings, not a new-subject '
              'blind test. Small sparse-box sample; no dense ADE, RPM/pose truth '
              'or realtime claim. Guard fallback is not event-sensing success.'))
    print('73 untuned temporal extension frozen',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources();return p
def run(case):
    verify();cfg=CASES[case]
    args=[sys.executable,str(ROOT/'core/run_force_innovation_forecast.py'),'--raw',cfg['raw'],'--output',str(OUT/case),
        '--start',str(cfg['start']),'--duration',str(cfg['duration']),'--verify-frames']
    if case!='negative5':args.append('--preview')
    with (OUT/(case+'_parent.log')).open('x',encoding='utf-8') as f:
        r=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert r.returncode==0,(case,r.returncode);check(case);print(case,'73 RAW extension checked',flush=True)
def check(case):
    verify();rr=rows(OUT/case/'observations_forecasts.jsonl');n=round(CASES[case]['duration']*50);assert len(rr)==n
    old=rows(previous.OUT/case/'observations_forecasts.jsonl');old72=rows(cache.OUT/case/'forecasts.jsonl')
    increments=['increment_guarded_events','increment_guarded_constants','increment_guarded_all']
    try:
        for r,o,e in zip(rr,old,old72):
            for k in ['t','geometry20','body_observation','observation','sensing','slot_audit','body_boxes','frame_sha256','event_fields_sha256','event_activity','body_axis','event_measurement_audit']:
                assert r[k]==o[k],(r['t'],k)
            actual=r['frontends']['delayed_feedback'];expected=e['frontends']['event_force']
            for k in ['coefficients','predictions']:
                for m in cache.MODELS:assert actual[k].get(m)==expected[k].get(m),(r['t'],k,m)
            assert actual['increment_diagnostic']==expected['increment_diagnostic'],(r['t'],'increment_parameters')
            for key,value in expected.items():
                if key not in ['coefficients','predictions','increment_diagnostic']:assert actual[key]==value,(r['t'],key)
    except Exception as error:
        save(OUT/case/'regression_failure73.json',dict(error=str(error),actual=r,expected=o,expected72=e));raise
    count=0;video=OUT/case/'forecasts.mp4'
    if video.is_file():
        cap=cv2.VideoCapture(str(video))
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(720,1280,3);count+=1
        cap.release();assert count==n
    save(OUT/case/'regression73.json',dict(frames=n,old_prefix_frames=len(old),original30_prefix_exact=True,
        motion_support_and_phase_optimizations_exact=True,video_frames_decoded=count,labels_read=False))
def score():
    verify()
    # Stage22 scorer consumes a simple frontend wrapper; this is a post-run
    # adapter only. Actual inference has already exited before any label read.
    scoring=OUT/'scoring';scoring.mkdir(exist_ok=False)
    for case in CASES:
        assert load(OUT/case/'regression73.json')['original30_prefix_exact']
        folder=scoring/case;folder.mkdir()
        with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as f:
            for r in rows(OUT/case/'observations_forecasts.jsonl'):
                f.write(json.dumps(dict(t=r['t'],frontends=dict(event_force=r['frontends']['delayed_feedback'])),allow_nan=False)+'\n')
    save(scoring/'protocol.json',load(OUT/'protocol.json'))
    spec=importlib.util.spec_from_file_location('score73_original22',ROOT/'research/stage22/study22.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=scoring;m.INPUT=scoring;m.CASES=list(CASES);m.FRONTENDS=['event_force'];m.MODELS=MODELS;m.verify=verify;m.evaluate=cache.prior.evaluate
    m.label_path=lambda c:base.BASE/'raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train6.json');m.score()
    analysis=load(scoring/'analysis.json');table=[]
    import numpy as np
    for case in NEW_START:
        for region in ['known_prefix','later_extension']:
            cutoff=NEW_START[case]+(.4 if region=='later_extension' else 0.)
            for h in [.1,.2,.4]:
                for model in MODELS:
                    dd=[d for d in analysis['details'] if d['case']==case and d['model']==model and d['nominal_h']==h and d['target_state']=='present'
                        and (d['target_t']>cutoff+1e-8 if region=='later_extension' else d['target_t']<=cutoff+1e-8)]
                    valid=[d for d in dd if d['forecast_valid']]
                    table.append(dict(case=case,region=region,model=model,nominal_h=h,targets=len(dd),valid=len(valid),
                        mean_fde_px=float(np.mean([d['error_px'] for d in valid])) if valid else None,
                        joint_success=sum(d.get('joint_success',False) for d in valid)))
    save(OUT/'temporal_scores73.json',dict(table=table,
        scope='Later-extension targets >new_anchor_start+.4; middle boundary targets retained in full scoring but excluded from split table. Development recording, not blind validation.'))
    for r in table:
        if r['nominal_h']==.4 and r['model'] in ['trusted_acc','increment_periodic','increment_guarded_events','increment_periodic_constant']:
            print(r,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','check','score']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action in ['run','check']:globals()[a.action](a.case)
    else:globals()[a.action]()
