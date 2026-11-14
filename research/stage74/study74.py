"""Parallel CV/physical IMM on the full frozen23s observations."""
from pathlib import Path
import argparse,datetime,importlib.util,json,sys,time
import numpy as np
from hybrid74 import *
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage73'),str(ROOT/'research/stage62'),str(ROOT/'research/stage22')]
import study73 as previous
from independent62 import Managed62,Selector
base=previous.base;load=base.load;save=base.save;sha=base.sha;rows=base.rows
OUT=ROOT.parent/'output/stage74_parallel_cv_physics_20261021';CASES=previous.CASES;MODELS=previous.MODELS+ADDED
def sources():
    result={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),Path(__file__).with_name('hybrid74.py'),Path(__file__).with_name('RELATED_WORK.md')]}
    for p in VENDOR.iterdir():
        if p.is_file():result[p.relative_to(ROOT).as_posix()]=sha(p)
    return result
def inputs():return {c:sha(previous.OUT/c/'observations_forecasts.jsonl') for c in CASES}
def freeze():
    previous.verify();assert not OUT.exists()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),input_hashes=inputs(),cases=CASES,
        label_hashes=load(previous.OUT/'protocol.json')['label_hashes'],parameters=PARAMS,controls=controls(),
        actual_upstream=snapshot,
        design='Full author FilterPy IMM interaction with two4D hypothesis filters: '
               'box CV and controlled maneuver. Preserve original33 outputs. '
               'Position once per hypothesis; current image features, previous '
               'learned force parameters/gains. No fixed delayed-score wait for '
               'current innovation likelihood. Five ablations plus CV/agile zero control.',
        scope='Previously development-seen recordings/labels. No held-out success '
              'claim; compare sparse truth and separately named dense detector '
              'self-consistency. Correct synthetic force is an oracle mechanism control.',
        regime_definition='Causal observed-motion proxy: CV velocity change over at '
              'least .16s up to .24s. Maneuver if turn>=25deg with both speeds>=50 '
              'or velocity change>=150px/s; steady if turn<=10deg and change<=50. '
              'Other/insufficient history unknown. Not independently annotated maneuver truth.'))
    print('74 dual IMM frozen',flush=True)
def verify():
    previous.verify();p=load(OUT/'protocol.json');assert p['source_hashes']==sources() and p['input_hashes']==inputs();return p
def infer():
    verify();sys.addaudithook(base.guard)
    for case in CASES:
        rr=rows(previous.OUT/case/'observations_forecasts.jsonl');managed=Managed62('axis_learn');selector=Selector();model=HybridForecast();folder=OUT/case;folder.mkdir(exist_ok=False)
        timing=[];weights={m:[] for m in ADDED}
        with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as f:
            for r in rr:
                q62,d=base.checked_step(managed,selector,r);q=r['frontends']['delayed_feedback']
                for m in base.MODELS:assert q['coefficients'].get(m)==q62['coefficients'].get(m)
                tick=time.perf_counter();new=model.step(q,r['observation'],r['body_axis'],managed.predictor.axis.quality);timing.append((time.perf_counter()-tick)*1000)
                for m in previous.MODELS:assert new['coefficients'].get(m)==q['coefficients'].get(m)
                for m in ADDED:
                    p=new['dual_branch_diagnostic'][m]['mode_probability']
                    if p is not None:weights[m].append(p[1])
                f.write(json.dumps(dict(t=r['t'],frontends=dict(event_force=new)),allow_nan=False)+'\n')
        save(folder/'completion.json',dict(frames=len(rr),old33_exact=True,labels_read=False,
            fusion6_banks_median_ms=float(np.median(timing)),fusion6_banks_p95_ms=float(np.percentile(timing,95)),
            mean_physics_probability={m:float(np.mean(w)) if w else None for m,w in weights.items()}));print(case,'74 dual banks complete',flush=True)
def prefix():
    verify();sys.addaudithook(base.guard);table=[]
    for case,n in [('side6',225),('train3',60)]:
        rr=rows(previous.OUT/case/'observations_forecasts.jsonl')[:n];expected=rows(OUT/case/'forecasts.jsonl')[:n]
        managed=Managed62('axis_learn');selector=Selector();model=HybridForecast()
        for r,e in zip(rr,expected):
            base.checked_step(managed,selector,r)
            q=model.step(r['frontends']['delayed_feedback'],r['observation'],r['body_axis'],managed.predictor.axis.quality)
            assert q==e['frontends']['event_force']
        table.append(dict(case=case,frames=n,full39_and_mode_states_exact=True))
    save(OUT/'prefix_checks.json',dict(table=table,actual_RAW=False));print('74 causal cache prefixes exact',flush=True)
def score():
    verify();assert (OUT/'prefix_checks.json').is_file()
    spec=importlib.util.spec_from_file_location('score74_original22',ROOT/'research/stage22/study22.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=OUT;m.INPUT=OUT;m.CASES=list(CASES);m.FRONTENDS=['event_force'];m.MODELS=MODELS;m.verify=verify;m.evaluate=prior.evaluate
    m.label_path=lambda c:base.BASE/'raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train6.json');m.score()
    new=load(OUT/'analysis.json');old=load(previous.OUT/'scoring/analysis.json');lookup={(r['case'],r['model'],r['nominal_h']):r for r in old['table']};count=0
    for r in new['table']:
        if r['model'] in previous.MODELS:assert r==lookup[r['case'],r['model'],r['nominal_h']];count+=1
        if r['nominal_h']==.4 and r['model'] in ['cv','trusted_acc','increment_periodic']+ADDED and r['case']!='negative5':print({k:r[k] for k in ['case','model','mean_fde_px','joint_success','valid_forecasts']},flush=True)
    save(OUT/'score_checks.json',dict(previous33_rows_exact=count,total_rows=len(new['table']),same_output_coverage=True))
def regimes():
    verify();details=load(OUT/'analysis.json')['details'];summaries=[];counts=[];dense=[]
    for case in CASES:
        rr=rows(previous.OUT/case/'observations_forecasts.jsonl');pred=rows(OUT/case/'forecasts.jsonl');tags={};history=[];tt=np.array([r['t'] for r in rr])
        for r in rr:
            q=r['frontends']['delayed_feedback'];tag='unknown'
            if q['valid']:
                v=np.asarray(q['coefficients']['cv']['velocity']);past=[x for x in history if .16-1e-8<=r['t']-x[0]<=.24+1e-8]
                if past:
                    old=past[-1][1];change=float(np.linalg.norm(v-old));speed=np.linalg.norm(v);old_speed=np.linalg.norm(old)
                    angle=float(np.degrees(np.arccos(np.clip(v@old/max(speed*old_speed,1e-9),-1,1))))
                    if change>=150 or (speed>=50 and old_speed>=50 and angle>=25):tag='maneuver_proxy'
                    elif change<=50 and angle<=10:tag='steady_proxy'
                history.append((r['t'],v));history=[x for x in history if r['t']-x[0]<=.24+1e-8]
            tags[round(r['t'],8)]=tag
        counts.append(dict(case=case,**{s:sum(v==s for v in tags.values()) for s in ['steady_proxy','maneuver_proxy','unknown']}))
        for regime in ['steady_proxy','maneuver_proxy','unknown']:
            for h in [.1,.2,.4]:
                for model in ['cv','trusted_acc','increment_periodic']+ADDED:
                    dd=[d for d in details if d['case']==case and d['model']==model and d['nominal_h']==h and d['forecast_valid'] and d['target_state']=='present' and tags[round(d['anchor_t'],8)]==regime]
                    summaries.append(dict(case=case,regime=regime,model=model,nominal_h=h,n=len(dd),mean_fde_px=float(np.mean([d['error_px'] for d in dd])) if dd else None))
        # This larger dense table explicitly uses future DETECTOR positions,
        # a self-consistency diagnostic, never independent accuracy truth.
        for r,p in zip(rr,pred):
            q=p['frontends']['event_force']
            if not q['valid']:continue
            for h in [.1,.2,.4]:
                j=int(np.searchsorted(tt,r['t']+h-1e-8))
                if j>=len(rr) or abs(tt[j]-r['t']-h)>.001 or not rr[j]['observation'].get('detected'):continue
                z=np.asarray(rr[j]['observation']['center'])
                for model in ['cv','trusted_acc','increment_periodic']+ADDED:
                    dense.append(dict(case=case,regime=tags[round(r['t'],8)],model=model,h=h,
                        error=float(np.linalg.norm(np.asarray(prior.evaluate(q['coefficients'][model],h))-z))))
    dense_summary=[]
    for case in CASES:
        for regime in ['steady_proxy','maneuver_proxy','unknown']:
            for h in [.1,.2,.4]:
                for model in ['cv','trusted_acc','increment_periodic']+ADDED:
                    dd=[d['error'] for d in dense if d['case']==case and d['regime']==regime and d['h']==h and d['model']==model]
                    dense_summary.append(dict(case=case,regime=regime,model=model,nominal_h=h,n=len(dd),mean_detector_consistency_error_px=float(np.mean(dd)) if dd else None))
    save(OUT/'regime_analysis74.json',dict(causal_regime_counts=counts,sparse_box_errors=summaries,
        dense_detector_self_consistency=dense_summary,regimes_independently_annotated=False,
        caveat='Regime labels are causal observed-CV proxies, not physical maneuver truth. Dense future detector errors are not independent trajectory accuracy.'))
    print('74 separate steady/maneuver diagnostics saved',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','infer','prefix','score','regimes']);globals()[p.parse_args().action]()
