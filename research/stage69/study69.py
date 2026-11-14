"""Preserve62 complete baseline; append event-force forecast ablations."""
from pathlib import Path
import argparse,datetime,importlib.util,json,sys
import numpy as np
from force69 import *
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage68'),str(ROOT/'research/stage64')]
import study68 as activity
import study64 as body
base=body.prior.previous
OUT=ROOT.parent/'output/stage69_event_force_forecast_20261021'
CASES=activity.CASES;load=base.load;save=base.save;sha=base.sha;rows=base.rows

def sources():
    result=base.sources()
    for p in [Path(__file__),Path(__file__).with_name('force69.py'),ROOT/'research/stage68/activity68.py']:result[p.relative_to(ROOT).as_posix()]=sha(p)
    return result

def inputs():
    return {c:dict(current=sha(base.REFERENCE/f'off_{c}/observations_forecasts.jsonl'),
        geometry=sha(body.OUT/c/'observations.jsonl'),activity=sha(activity.OUT/c/'activity.jsonl'),
        baseline=sha(base.OUT/c/'forecasts.jsonl')) for c in CASES}

def freeze():
    activity.verify();body.verify();assert not OUT.exists()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=CASES,
        source_hashes=sources(),input_hashes=inputs(),parameters=PARAMS,controls=controls(),
        label_hashes=load(base.OUT/'protocol.json')['label_hashes'],
        design='Unchanged62 full15 + eight added force/residual branches. Every activity '
               'branch has identical-direction, identical-admission constant-input control. '
               'Same trusted-history fallback, same output coverage. Current observed '
               'position feedback only after delay. Save all forecasts before label scoring.',
        scope='Development-recording cache ablation, no new joint RAW deployment or '
              'independent RPM/pose truth. Effective2D coefficients cannot identify true mass/wind.'))
    print('69 protocol frozen',flush=True)

def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources() and p['input_hashes']==inputs();return p

def infer():
    verify();sys.addaudithook(base.guard);summary=[]
    for case in CASES:
        rr=rows(base.REFERENCE/f'off_{case}/observations_forecasts.jsonl');bb=rows(body.OUT/case/'observations.jsonl')
        aa=rows(activity.OUT/case/'activity.jsonl');old=rows(base.OUT/case/'forecasts.jsonl')
        managed=Managed62('axis_learn');selector=Selector();force=ForceForecast();folder=OUT/case;folder.mkdir(exist_ok=False)
        counts={m:dict(active=0,launch=0,changed_vs_history=0) for m in ADDED}
        with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as f:
            for r,b,a,ref in zip(rr,bb,aa,old):
                assert r['t']==b['t']==a['t']==ref['t']
                q,d=base.checked_step(managed,selector,r)
                assert q==ref['frontends']['axis_learn'] and d==ref['lifecycle']['axis_learn'],(case,r['t'],'baseline62')
                new=force.step(q,r['observation'],b['methods']['aligned_total'],a['activity'],managed.predictor.axis.quality)
                for m in ORIGINAL_MODELS:assert new['coefficients'].get(m)==q['coefficients'].get(m)
                for m,diag in new['event_force_diagnostic']['models'].items():
                    counts[m]['active']+=diag['active'];counts[m]['launch']+=diag['launch']
                    counts[m]['changed_vs_history']+=bool(new['valid'] and new['coefficients'][m]!=q['coefficients']['trusted_acc'])
                f.write(json.dumps(dict(t=r['t'],frontends=dict(event_force=new),lifecycle=d),allow_nan=False)+'\n')
        save(folder/'completion.json',dict(frames=len(rr),baseline_full15_exact62=True,labels_read=False,counts=counts,
            updates={m:force.models[m].updates for m in ADDED}))
        summary.append(dict(case=case,frames=len(rr),counts=counts));print(case,counts,flush=True)
    save(OUT/'inference_checks.json',dict(table=summary,baseline_full15_exact62=True,all_predictions_saved_before_scoring=True))

def prefix():
    verify();sys.addaudithook(base.guard);checks=[]
    for case,n in [('side6',225),('train3',60)]:
        rr=rows(base.REFERENCE/f'off_{case}/observations_forecasts.jsonl')[:n];bb=rows(body.OUT/case/'observations.jsonl')[:n]
        aa=rows(activity.OUT/case/'activity.jsonl')[:n];expected=rows(OUT/case/'forecasts.jsonl')[:n]
        managed=Managed62('axis_learn');selector=Selector();force=ForceForecast()
        for r,b,a,e in zip(rr,bb,aa,expected):
            q,d=base.checked_step(managed,selector,r);new=force.step(q,r['observation'],b['methods']['aligned_total'],a['activity'],managed.predictor.axis.quality)
            assert new==e['frontends']['event_force'] and d==e['lifecycle']
        checks.append(dict(case=case,frames=n,full23_forecast_and_learning_exact=True))
    save(OUT/'prefix_checks.json',dict(table=checks,scope='Independent causal cache prefixes, not new RAW joint entry.'))
    print('69 prefix checks passed',flush=True)

def score():
    verify();assert (OUT/'prefix_checks.json').is_file()
    spec=importlib.util.spec_from_file_location('score69_original22',ROOT/'research/stage22/study22.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=OUT;m.INPUT=OUT;m.CASES=list(CASES);m.FRONTENDS=['event_force'];m.MODELS=MODELS;m.verify=verify;m.evaluate=prior.evaluate
    m.label_path=lambda c:base.BASE/'raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train6.json')
    m.score();new=load(OUT/'analysis.json');old=load(base.OUT/'analysis.json')
    lookup={(r['case'],r['model'],r['nominal_h']):r for r in old['table'] if r['frontend']=='axis_learn'}
    count=0
    for r in new['table']:
        ref=lookup[r['case'],r['model'] if r['model'] in ORIGINAL_MODELS else 'trusted_acc',r['nominal_h']]
        if r['model'] in ORIGINAL_MODELS:assert r==dict(ref,frontend='event_force');count+=1
        for k in ['present_targets','anchors_available','valid_forecasts','absent_targets','known_negative_published_anchors']:assert r[k]==ref[k]
        if r['nominal_h']==.4 and r['case']!='negative5' and r['model'] in ['trusted_acc']+ADDED:print({k:r[k] for k in ['case','model','mean_fde_px','joint_success']},flush=True)
    save(OUT/'score_checks.json',dict(rows=len(new['table']),baseline_rows_exact62=count,all_coverage_exact=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','infer','prefix','score']);globals()[p.parse_args().action]()
