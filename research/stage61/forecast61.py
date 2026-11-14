"""Frozen actual RAW/full15 controls: stable baseline vs stable paired observations."""
from pathlib import Path
import argparse,datetime,hashlib,inspect,json,sys
import cv2,numpy as np
import study61 as current
import runtime_full61
from rotor_worker61 import worker_source
ROOT=current.ROOT;sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT));import run_stable_paired_forecast as runner
OUT=current.OUT/'forecast';REF=current.isolated60.REF
load=current.load;save=current.save;sha=current.sha;rows=current.rows
HISTORY=['hold','cv','cv_hist_acc','bounded_acc','huber_acc','trusted_acc']
CASES={f'{mode}_{c}':dict(current.isolated60.CASES[c],line_mode=mode) for mode in ['off','paired'] for c in ['clear6','side6','train3','negative5']}
SCORING=list(CASES)
CASES.update(pilot_off_clear6=dict(CASES['off_clear6'],duration=.8),
    prefix_off_side6=dict(CASES['off_side6'],duration=4.5),prefix_paired_side6=dict(CASES['paired_side6'],duration=4.5),
    prefix_off_train3=dict(CASES['off_train3'],duration=1.2),prefix_paired_train3=dict(CASES['paired_train3'],duration=1.2))
SCHEDULE=['pilot_off_clear6']+[f'{mode}_{c}' for c in ['clear6','side6','train3','negative5'] for mode in ['off','paired']]+['prefix_off_side6','prefix_paired_side6','prefix_off_train3','prefix_paired_train3']

def sources():
    r=current.sources()
    for p in [Path(__file__),ROOT/'core/run_stable_paired_forecast.py',ROOT/'core/run_paired_forecast.py',ROOT/'research/stage60/rotor_worker60.py',*[Path(__file__).with_name(n) for n in ['rotor_worker61.py','runtime_full61.py']]]:r[p.relative_to(ROOT).as_posix()]=sha(p)
    return r
def freeze():
    current.verify();runtime=runtime_full61.verify();assert not OUT.exists()
    assert all(r['RNG_intervention_exact'] for r in load(current.OUT/'results.json')['table'])
    compile(worker_source(),'<61 stable worker>','exec')
    for mode in ['off','paired']:
        namespace=runner.build(mode,32)
        try:assert len(namespace['MODELS'])==15 and namespace['METHODS']==['delayed_feedback']
        finally:namespace.clear()
    old=load(REF/'protocol.json')
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,label_hashes=old['label_hashes'],
       runtime=runtime,derived_parent_sha256=hashlib.sha256(runner.pipeline_source().encode()).hexdigest(),derived_worker_sha256=hashlib.sha256(worker_source().encode()).hexdigest(),
       design='Two live RAW-only shared-decode full15 modes, both61 preline active. Off disables only60 pair/profile/fusion, never reverts to degenerate ellipse baseline. Same20ms body NN/current images, original10ms full frequency buffer/ReplaceStale/RLS/selector. Corrected rotor geometry may differ57; body/six-history/coverage must remain exactly57. No seed changes in inference.',
       verification='No-thin clear0.8s off pilot full15/lifecycle exactly57; early current observations where pilot context matches exactly module61. Independent side4.5s and train31.2s RAW prefix in BOTH modes: all current state, ROI/preline/line audit, named event hashes and full15 forecasts/lifecycle exact. All native10/full videos counted. Predictions complete before original22 sparse future scoring.',
       schedule=SCHEDULE,scoring_cases=SCORING,
       prior_art=['https://docs.opencv.org/4.x/db/d73/classcv_1_1LineSegmentDetector.html','https://github.com/event-driven-robotics/RT-EvLDT'],
       actual_reuse='Frozen61 preline AST and existing geometry._line, frozen60 pairing/profile/fusion,57 candidate gate,54 shared packets,53/52/33 complete physics/frequency/state and55/original22 score. OpenCV actual5.0 binaries pinned empirically;4.x docs not used to explain5.0 RNG implementation.',
       scope='Development recordings and sparse visible-box endpoint error only; no true COM/RPM/attitude labels or realtime claim. No parameter fitting to new scores, no default promotion before complete comparison.'))
    (OUT/'derived_parent61.py').write_text(runner.pipeline_source(),encoding='utf-8');(OUT/'derived_worker61.py').write_text(worker_source(),encoding='utf-8');print('61 full RAW stable baseline/paired protocol frozen',flush=True)
def verify():
    current.verify();p=load(OUT/'protocol.json');assert p['source_hashes']==sources() and p['runtime']==runtime_full61.verify()
    assert p['derived_parent_sha256']==hashlib.sha256(runner.pipeline_source().encode()).hexdigest() and p['derived_worker_sha256']==hashlib.sha256(worker_source().encode()).hexdigest()
    return p
def run(case):
    verify();cfg=CASES[case];runner.run(cfg['raw'],OUT/case,cfg['start'],cfg['duration'],32,not case.startswith('prefix_'),True,cfg['line_mode'])
def failure(case,t,key,actual,expected):
    save(OUT/case/'regression_failure61.json',dict(t=t,key=key,actual=actual,expected=expected));raise AssertionError((case,t,key,'exact failure retained'))
def audit(case):
    verify();cfg=CASES[case];ref=cfg['reference'];new=rows(OUT/case/'observations_forecasts.jsonl');old=rows(REF/ref/'observations_forecasts.jsonl');ticks=rows(OUT/case/'rotor_ticks.jsonl');completion=load(OUT/case/'completion.json');native=load(OUT/case/'rotor_completion.json')
    assert completion['source_hashes_unchanged'] and native['source_hashes_unchanged'] and not completion['labels_read'] and not native['labels_read']
    assert native['opencv_runtime']==runtime_full61.verify() and native['opencv_threads']==1
    assert len(new)==round(cfg['duration']*50) and len(ticks)==2*len(new)
    for a,b in zip(new,old):
        for name in ['t','event_count','filtered_event_count','geometry20','body_observation','body_boxes','frame_sha256','body_hint_sent']:
            if a[name]!=b[name]:failure(case,a['t'],name,a[name],b[name])
        qa=a['frontends']['delayed_feedback'];qb=b['frontends']['delayed_feedback'];assert qa['valid']==qb['valid']
        if qa['valid']:
            assert all(n in qa['coefficients'] and n in qb['coefficients'] for n in HISTORY)
            for name in HISTORY:
                if qa['coefficients'][name]!=qb['coefficients'][name]:failure(case,a['t'],name,qa['coefficients'][name],qb['coefficients'][name])
        else:assert qa['coefficients']==qb['coefficients']=={}
        if case=='pilot_off_clear6':
            if a['frontends']!=b['frontends']:failure(case,a['t'],'off pilot full15',a['frontends'],b['frontends'])
            assert a['lifecycle']==b['lifecycle']
    # Replay the same initial current module context; side61 pilot starts at112,
    # unlike formal108, so no false same-context assertion is made there.
    checked=0
    if ref in ['clear6','train3','negative5']:
        module=load(current.OUT/f'normal_{ref}.json')['records'];key='stable' if cfg['line_mode']=='off' else 'paired'
        for a,b in zip(ticks,module):
            if a['observation']!=b[key]:failure(case,a['t'],'same-context module61 observation',a['observation'],b[key])
            assert a['roi_diagnostic']['preline_audit']==b['stable_preline' if cfg['line_mode']=='off' else 'paired_preline'];checked+=1
    oldticks=rows(REF/ref/'rotor_ticks.jsonl')
    for a,b in zip(ticks,oldticks):
        assert a['t']==b['t'] and a['event_count']==b['event_count'] and a['geometry_event_count']==a['event_count']
        assert a['roi_diagnostic']['geometry_events']==a['event_count']
    video_frames=0
    if (OUT/case/'forecasts.mp4').exists():
        cap=cv2.VideoCapture(str(OUT/case/'forecasts.mp4'))
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape[:2]==(720,1280);video_frames+=1
        cap.release();assert video_frames==len(new)
    summary=dict(case=case,parent=len(new),native10=len(ticks),video_decoded_frames=video_frames,body_six_history_coverage_exact57=True,
       pilot_complete15_exact57=case=='pilot_off_clear6',same_context_module_current_ticks_exact=checked,
       preline_routed=sum(a['roi_diagnostic']['preline_audit']['thin_routed'] for a in ticks),
       preline_rejected=sum(a['roi_diagnostic']['preline_audit']['thin_rejected'] for a in ticks),
       line_measured=sum(a['roi_diagnostic']['preline_audit']['line_measured'] for a in ticks),
       inserted=sum(a['roi_diagnostic']['line_audit'].get('inserted',0) for a in ticks),fused=sum(a['roi_diagnostic']['line_audit'].get('fused',0) for a in ticks),
       four_visible=sum(a['sensing']['four_visible'] for a in ticks),four_frequencies=sum(a['sensing']['four_frequencies'] for a in ticks),all_current_events_preserved=True)
    save(OUT/case/'regression61.json',summary);print(summary,flush=True)
def prefix():
    verify();result=[]
    for mode in ['off','paired']:
        for ref,nparents in [('side6',225),('train3',60)]:
            name=f'prefix_{mode}_{ref}';a=rows(OUT/name/'observations_forecasts.jsonl');b=rows(OUT/f'{mode}_{ref}'/'observations_forecasts.jsonl');assert len(a)==nparents
            for x,y in zip(a,b):
                for key in ['t','event_count','filtered_event_count','geometry20','body_observation','observation','sensing','slot_audit','body_boxes','frame_sha256','frontends','lifecycle','event_fields_sha256','event_packet_nbytes','body_hint_sent','roi_diagnostic']:
                    if x[key]!=y[key]:failure(name,x['t'],'prefix '+key,x[key],y[key])
            a=rows(OUT/name/'rotor_ticks.jsonl');b=rows(OUT/f'{mode}_{ref}'/'rotor_ticks.jsonl');assert len(a)==2*nparents
            for x,y in zip(a,b):
                for key in ['t','event_count','geometry_event_count','observation','sensing','slot_audit','roi_diagnostic']:
                    if x[key]!=y[key]:failure(name,x['t'],'native prefix '+key,x[key],y[key])
            result.append(dict(mode=mode,case=ref,parent=nparents,native10=2*nparents))
    save(OUT/'raw_prefix_checks.json',dict(table=result,parent_ticks=570,native10_ticks=1140,actual_RAW_prefix=True,all_states_full15_and_named_event_hashes_exact=True,transport_byte_checksum_preserved=True));print('61 two-mode side/train3 RAW prefixes all exact',flush=True)
def score():
    verify()
    for case in CASES:assert load(OUT/case/'regression61.json')['body_six_history_coverage_exact57']
    assert load(OUT/'raw_prefix_checks.json')['all_states_full15_and_named_event_hashes_exact']
    old55=current.previous.previous.previous.previous.previous
    namespace=dict(old55.__dict__);namespace.update(OUT=OUT,CASES=CASES,SCORING=SCORING,verify=verify)
    exec(compile(inspect.getsource(old55.score),'<61 unchanged55 original22 sparse score>','exec'),namespace);namespace['score']()
    new=load(OUT/'analysis.json');old=load(REF/'analysis.json');lookup={(r['case'],r['model'],r['nominal_h']):r for r in old['table']};exact=0;comparison=[]
    bynew={(r['case'],r['model'],r['nominal_h']):r for r in new['table']}
    for r in new['table']:
        ref=CASES[r['case']]['reference'];b=lookup[ref,r['model'],r['nominal_h']]
        for key in ['present_targets','anchors_available','valid_forecasts','absent_targets','known_negative_published_anchors']:assert r[key]==b[key]
        if r['model'] in HISTORY:
            expected=dict(b,case=r['case']);assert r==expected;exact+=1
    for ref in ['clear6','side6','train3']:
        for model in ['trusted_acc','adaptive_axis_freq','adaptive_all_mix']:
            for h in [.1,.2,.4]:
                a=lookup[ref,model,h];b=bynew[f'off_{ref}',model,h];c=bynew[f'paired_{ref}',model,h]
                comparison.append(dict(case=ref,model=model,horizon=h,legacy57_fde_px=a['mean_fde_px'],stable_fde_px=b['mean_fde_px'],paired_fde_px=c['mean_fde_px'],valid_forecasts=b['valid_forecasts'],present_targets=b['present_targets'],legacy_joint_success=a['joint_success'],stable_joint_success=b['joint_success'],paired_joint_success=c['joint_success']))
    save(OUT/'score_checks.json',dict(rows=len(new['table']),history_rows_exact57=exact,all_coverage_exact57=True,comparison=comparison));print(comparison,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','audit','prefix','score']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action in ['run','audit']:globals()[a.action](a.case)
    else:globals()[a.action]()
