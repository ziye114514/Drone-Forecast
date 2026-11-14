"""Full actual RAW frequency/15-model validation conditional on changed geometry."""
from pathlib import Path
import argparse,datetime,hashlib,html,inspect,json,sys
import cv2,numpy as np
import study60 as current
import raw60
import isolated60
ROOT=current.ROOT;sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT));import run_paired_forecast as runner
from rotor_worker60 import worker_source
REF=current.previous.previous.previous.OUT
OUT=current.OUT/'forecast'
CASES={c:dict(current.previous.previous.previous.CASES[c],line_mode='paired') for c in current.CASES}
CASES.update(pilot_off_clear6=dict(CASES['clear6'],duration=.8,line_mode='off'),prefix_side6=dict(CASES['side6'],duration=4.5))
SCORING=list(current.CASES);load=current.load;save=current.save;sha=current.sha;rows=current.previous.rows

def sources():
    r=isolated60.sources()
    for p in [Path(__file__),ROOT/'core/run_paired_forecast.py',Path(__file__).with_name('rotor_worker60.py')]:r[p.relative_to(ROOT).as_posix()]=sha(p)
    return r

def freeze():
    isolated60.verify();assert not OUT.exists();r=load(isolated60.OUT/'raw_summary.json')
    assert r['off_exact57'] and any(x['changed'] for x in r['table']),'No change: do not rerun identical full forecasts'
    compile(worker_source(),'<60 worker>','exec')
    for mode in ['off','paired']:
        namespace=runner.build(mode,32)
        try:assert len(namespace['MODELS'])==15 and namespace['METHODS']==['delayed_feedback']
        finally:namespace.clear()
    original=load(REF/'protocol.json')
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,label_hashes=original['label_hashes'],
       derived_parent_sha256=hashlib.sha256(runner.pipeline_source().encode()).hexdigest(),derived_worker_sha256=hashlib.sha256(worker_source().encode()).hexdigest(),
       design='Live actual RAW single shared decode, original20ms body NN and complete15 models/10ms frequency/ReplaceStale/RLS/delayed selector. Only side-line proposals/fusion changed. Off pilot must exactly original57 all15, live geometry must match standalone current-only RAW60, full native20 body/six history/coverage must exactly57. Independent4.5s actual RAW prefix must exactly new full side including all semantic named-field SHA and states. All predictions saved before sparse truth scoring.',
       parameters='Exact frozen pairs60/profile60/frontend60. Off flag disables line adapter but keeps original57 candidate ROI; internal --geometry-roi is transport only, not an off-ROI ablation.',
       schedule=['pilot_off_clear6']+SCORING+['prefix_side6'],
       scope='Development clips, sparse visible-box endpoints not COM or dense ADE. No actual RPM/attitude truth. New line confidence is a heuristic and not calibrated probability; no promotion until complete comparison.'))
    (OUT/'derived_parent.py').write_text(runner.pipeline_source(),encoding='utf-8');(OUT/'derived_worker.py').write_text(worker_source(),encoding='utf-8')
    print('60 full RAW future controls frozen',flush=True)

def verify():
    isolated60.verify();p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    assert p['derived_parent_sha256']==hashlib.sha256(runner.pipeline_source().encode()).hexdigest() and p['derived_worker_sha256']==hashlib.sha256(worker_source().encode()).hexdigest()
    return p

def run(case):
    verify();cfg=CASES[case];runner.run(cfg['raw'],OUT/case,cfg['start'],cfg['duration'],32,case!='prefix_side6',True,cfg['line_mode'])

def audit(case):
    verify();cfg=CASES[case];ref=cfg['reference'];new=rows(OUT/case/'observations_forecasts.jsonl');old=rows(REF/ref/'observations_forecasts.jsonl')
    ticks=rows(OUT/case/'rotor_ticks.jsonl');old_ticks=rows(REF/ref/'rotor_ticks.jsonl');completion=load(OUT/case/'completion.json');native=load(OUT/case/'rotor_completion.json')
    assert completion['source_hashes_unchanged'] and native['source_hashes_unchanged'] and not native['labels_read']
    assert len(new)==round(cfg['duration']*50) and len(ticks)==len(new)*2
    history=['hold','cv','cv_hist_acc','bounded_acc','huber_acc','trusted_acc']
    for a,b in zip(new,old):
        for name in ['t','event_count','filtered_event_count','geometry20','body_observation','body_boxes','frame_sha256']:assert a[name]==b[name],(case,a['t'],name)
        qa=a['frontends']['delayed_feedback'];qb=b['frontends']['delayed_feedback'];assert qa['valid']==qb['valid']
        for name in history:assert qa['coefficients'][name]==qb['coefficients'][name],(case,a['t'],name)
        if cfg['line_mode']=='off':assert a['frontends']==b['frontends'] and a['lifecycle']==b['lifecycle']
    standalone=load(isolated60.OUT/f'raw_{ref}.json')['records']
    for i,(a,b) in enumerate(zip(ticks,old_ticks)):
        assert a['event_count']==b['event_count'] and a['geometry_event_count']==a['event_count']
        expected=standalone[i]['off' if cfg['line_mode']=='off' else 'paired'];assert a['observation']==expected,(case,a['t'],'live frontend differs standalone RAW')
        if cfg['line_mode']=='off':assert a['observation']==b['observation'] and a['sensing']==b['sensing'] and a['slot_audit']==b['slot_audit']
    count=0
    if (OUT/case/'forecasts.mp4').exists():
        cap=cv2.VideoCapture(str(OUT/case/'forecasts.mp4'))
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape[:2]==(720,1280);count+=1
        cap.release();assert count==len(new)
    d=dict(case=case,parent=len(new),native10=len(ticks),video_decoded_frames=count,body_six_history_and_coverage_exact57=True,
       live_geometry_exact_standalone_RAW60=True,off_complete15_exact57=cfg['line_mode']=='off',
       inserted=sum(a['roi_diagnostic']['line_audit'].get('inserted',0) for a in ticks),fused=sum(a['roi_diagnostic']['line_audit'].get('fused',0) for a in ticks),
       four_visible=sum(a['sensing']['four_visible'] for a in ticks),four_frequencies=sum(a['sensing']['four_frequencies'] for a in ticks))
    save(OUT/case/'regression60.json',d);print(d,flush=True)

def prefix():
    verify();a=rows(OUT/'prefix_side6/observations_forecasts.jsonl');b=rows(OUT/'side6/observations_forecasts.jsonl');assert len(a)==225
    for x,y in zip(a,b):
        for key in ['t','event_count','filtered_event_count','geometry20','body_observation','observation','sensing','slot_audit','body_boxes','frame_sha256','frontends','lifecycle','event_fields_sha256','event_packet_nbytes','body_hint_sent','roi_diagnostic']:assert x[key]==y[key],(x['t'],key)
    a=rows(OUT/'prefix_side6/rotor_ticks.jsonl');b=rows(OUT/'side6/rotor_ticks.jsonl');assert len(a)==450
    for x,y in zip(a,b):
        for key in ['t','event_count','geometry_event_count','observation','sensing','slot_audit','roi_diagnostic']:assert x[key]==y[key],(x['t'],key)
    save(OUT/'raw_prefix_checks.json',dict(parent_ticks=225,native10_ticks=450,actual_RAW_prefix=True,all_states_and_semantic_named_event_fields_exact=True,transport_byte_checksum_preserved=True));print('60 full RAW prefix exact',flush=True)

def score():
    verify()
    for case in CASES:assert load(OUT/case/'regression60.json')['live_geometry_exact_standalone_RAW60']
    assert load(OUT/'raw_prefix_checks.json')['all_states_and_semantic_named_event_fields_exact']
    old55=current.previous.previous.previous.previous
    namespace=dict(old55.__dict__);namespace.update(OUT=OUT,CASES=CASES,SCORING=SCORING,verify=verify)
    exec(compile(inspect.getsource(old55.score),'<60 unchanged55 original22 scorer>','exec'),namespace);namespace['score']()
    new=load(OUT/'analysis.json');old=load(REF/'analysis.json');ref={(r['case'],r['model'],r['nominal_h']):r for r in old['table']};exact=0;comparison=[]
    for r in new['table']:
        b=ref[r['case'],r['model'],r['nominal_h']]
        for key in ['present_targets','anchors_available','valid_forecasts','absent_targets','known_negative_published_anchors']:assert r[key]==b[key],(r['case'],key)
        if r['model'] in ['hold','cv','cv_hist_acc','bounded_acc','huber_acc','trusted_acc']:assert r==b;exact+=1
        if r['nominal_h']==.4 and r['case']!='negative5' and r['model'] in ['trusted_acc','adaptive_axis_freq','adaptive_all_mix']:
            comparison.append(dict(case=r['case'],model=r['model'],baseline_fde_px=b['mean_fde_px'],paired_fde_px=r['mean_fde_px'],valid_forecasts=r['valid_forecasts'],present_targets=r['present_targets'],baseline_joint_success=b['joint_success'],paired_joint_success=r['joint_success']))
    save(OUT/'score_checks.json',dict(rows=len(new['table']),history_rows_exact57=exact,coverage_exact57=True,comparison=comparison));print(comparison,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','audit','prefix','score']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action in ['run','audit']:globals()[a.action](a.case)
    else:globals()[a.action]()
