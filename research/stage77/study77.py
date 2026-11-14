"""Fixed structural factor ablations; retain original39 forecasts exactly."""
from pathlib import Path
import argparse,datetime,importlib.util,json,sys,time
import numpy as np
from separated77 import *
sys.path[:0]=[str(ROOT/'research/stage76'),str(ROOT/'research/stage75'),str(ROOT/'research/stage22')]
import diagnose76 as audit
import study75 as actual
base=actual.cache
OUT=ROOT.parent/'output/stage77_separated_filter_forecast_20261021'
CASES=base.CASES;MODELS=base.MODELS+ADDED
load=base.load;save=base.save;sha=base.sha;rows=base.rows
def sources():
    files=[Path(__file__),Path(__file__).with_name('separated77.py'),Path(__file__).with_name('RELATED_WORK.md')]
    files.extend(p for p in VENDOR.iterdir() if p.is_file())
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files}
def inputs():return {c:sha(actual.OUT/c/'observations_forecasts.jsonl') for c in CASES}
def freeze():
    actual.verify();base.verify();assert not OUT.exists()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=CASES,
        source_hashes=sources(),input_hashes=inputs(),parameters=PARAMS,controls=controls(),upstream=SNAPSHOT,
        audit_hash=sha(audit.OUT/'audit76.json'),label_hashes=load(base.OUT/'protocol.json')['label_hashes'],
        design='Ten predeclared structural arms: refresh-only, history-when-missing, '
               'IMM/MMAE x state/forecast-only controls with current support, plus '
               'matched constant variants. All previous39 preserved, same output '
               'coverage. Refresh-only filtered state/covariance/mode probability '
               'must match74 exactly; only future history input changes.',
        scope='Development-seen data and hypotheses informed by prior errors. '
              'No held-out or parameter-search claim. Save predictions before '
              'separate sparse future-box scoring.'))
    print('77 structural protocol frozen',flush=True)
def verify():
    actual.verify();base.verify();p=load(OUT/'protocol.json');assert p['source_hashes']==sources() and p['input_hashes']==inputs();return p
def infer():
    verify();sys.addaudithook(base.base.guard)
    for case in CASES:
        rr=rows(actual.OUT/case/'observations_forecasts.jsonl');model=SeparatedForecast();folder=OUT/case;folder.mkdir(exist_ok=False)
        timing=[];current=0;fallback=0
        with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as f:
            for r in rr:
                q=r['frontends']['delayed_feedback'];tick=time.perf_counter();new=model.step(q,r['observation']);timing.append((time.perf_counter()-tick)*1000)
                for m in base.MODELS:assert new['coefficients'].get(m)==q['coefficients'].get(m)
                d=new['separated_diagnostic'];current+=d['current'];fallback+=bool(new['valid'] and not d['current'])
                if q['valid']:
                    c=new['coefficients']['rollout_refresh'];old=q['coefficients']['dual_cv_event_increment']
                    assert c['position']==old['position'] and c['velocity']==old['velocity'],(case,r['t'],'refresh state')
                    diag=d['models']['rollout_refresh'];ref=q['dual_branch_diagnostic']['dual_cv_event_increment']
                    assert diag['mode_probability']==ref['mode_probability'] and diag['position_covariance']==ref['position_covariance'],(case,r['t'],'refresh probabilities/covariance')
                    if not d['current']:
                        for m in SPECS:assert new['coefficients'][m]==q['coefficients']['trusted_acc']
                f.write(json.dumps(dict(t=r['t'],frontends=dict(event_force=new)),allow_nan=False)+'\n')
        save(folder/'completion.json',dict(frames=len(rr),old39_exact=True,refresh_filter_state_exact74=True,
            current_supported=current,history_fallback_ticks=fallback,labels_read=False,
            inference10_arms_median_ms=float(np.median(timing)),inference10_arms_p95_ms=float(np.percentile(timing,95))))
        print(case,'77 complete',current,'current /',len(rr),flush=True)
def prefix():
    verify();sys.addaudithook(base.base.guard);table=[]
    for case,n in [('side6',225),('train3',60)]:
        rr=rows(actual.OUT/case/'observations_forecasts.jsonl')[:n];expected=rows(OUT/case/'forecasts.jsonl')[:n];model=SeparatedForecast()
        for r,e in zip(rr,expected):
            q=model.step(r['frontends']['delayed_feedback'],r['observation']);assert q==e['frontends']['event_force']
        table.append(dict(case=case,frames=n,all49_forecasts_and_states_exact=True))
    save(OUT/'prefix_checks.json',dict(table=table,actual_RAW_prefix=False));print('77 causal cache prefixes exact',flush=True)
def score():
    verify();assert (OUT/'prefix_checks.json').is_file()
    spec=importlib.util.spec_from_file_location('score77_original22',ROOT/'research/stage22/study22.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=OUT;m.INPUT=OUT;m.CASES=list(CASES);m.FRONTENDS=['event_force'];m.MODELS=MODELS;m.verify=verify;m.evaluate=prior.evaluate
    m.label_path=lambda c:base.base.BASE/'raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train6.json');m.score()
    new=load(OUT/'analysis.json');old=load(base.OUT/'analysis.json');lookup={(r['case'],r['model'],r['nominal_h']):r for r in old['table']};count=0
    for r in new['table']:
        reference=lookup[r['case'],r['model'] if r['model'] in base.MODELS else 'trusted_acc',r['nominal_h']]
        if r['model'] in base.MODELS:assert r==reference;count+=1
        for k in ['present_targets','anchors_available','valid_forecasts','absent_targets','known_negative_published_anchors']:assert r[k]==reference[k]
        if r['case']!='negative5' and r['nominal_h']==.4 and r['model'] in ['cv','trusted_acc','dual_cv_event_increment']+ADDED:
            print({k:r[k] for k in ['case','model','mean_fde_px','joint_success']},flush=True)
    save(OUT/'score_checks.json',dict(previous39_rows_exact=count,total_rows=len(new['table']),same_output_coverage=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','infer','prefix','score']);globals()[p.parse_args().action]()
