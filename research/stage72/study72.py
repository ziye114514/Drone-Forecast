"""Increment-only physics ablation on frozen current inputs."""
from pathlib import Path
import argparse,datetime,importlib.util,json,sys
from increment72 import *
sys.path[:0]=[str(ROOT/'research/stage70'),str(ROOT/'research/stage22')]
import study70 as previous
base=previous.base;activity=previous.previous.activity;body=previous.previous.body
load=base.load;save=base.save;sha=base.sha;rows=base.rows
OUT=ROOT.parent/'output/stage72_force_innovation_20261021';CASES=previous.CASES
MODELS=previous.MODELS+ADDED
def sources():return {p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),Path(__file__).with_name('increment72.py')]}
def inputs():return {c:sha(previous.OUT/c/'forecasts.jsonl') for c in CASES}
def freeze():
    previous.verify();assert not OUT.exists()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=CASES,
        source_hashes=sources(),input_hashes=inputs(),parameters=PARAMS,controls=controls(),
        label_hashes=load(previous.OUT/'protocol.json')['label_hashes'],
        scope='Known development clips; fixed prior parameters before this run. '
              'Previous results informed force-increment design, so not held-out '
              'validation. Future targets read only after predictions saved.'))
    print('72 increment protocol frozen',flush=True)
def verify():
    previous.verify();p=load(OUT/'protocol.json');assert p['source_hashes']==sources() and p['input_hashes']==inputs();return p
def infer():
    verify();sys.addaudithook(base.guard)
    for case in CASES:
        rr=rows(base.REFERENCE/f'off_{case}/observations_forecasts.jsonl');aa=rows(activity.OUT/case/'activity.jsonl');bb=rows(body.OUT/case/'observations.jsonl');old=rows(previous.OUT/case/'forecasts.jsonl')
        managed=previous.previous.Managed62('axis_learn');selector=previous.previous.Selector();increment=IncrementForecast()
        folder=OUT/case;folder.mkdir(exist_ok=False);counts={m:0 for m in ADDED}
        with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as f:
            for r,a,b,e in zip(rr,aa,bb,old):
                q,d=base.checked_step(managed,selector,r)
                new=increment.step(e['frontends']['event_force'],r['observation'],b['methods']['aligned_total'],a['activity'],managed.predictor.axis.quality)
                for m in previous.MODELS:assert new['coefficients'].get(m)==e['frontends']['event_force']['coefficients'].get(m)
                for m in ADDED:counts[m]+=bool(new['valid'] and new['coefficients'][m]!=new['coefficients']['trusted_acc'])
                f.write(json.dumps(dict(t=r['t'],frontends=dict(event_force=new)),allow_nan=False)+'\n')
        save(folder/'completion.json',dict(frames=len(rr),counts=counts,updates={m:increment.models[m].updates for m in ADDED},old26_exact=True,labels_read=False));print(case,counts,flush=True)
def score():
    verify();spec=importlib.util.spec_from_file_location('score72_original22',ROOT/'research/stage22/study22.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=OUT;m.INPUT=OUT;m.CASES=list(CASES);m.FRONTENDS=['event_force'];m.MODELS=MODELS;m.verify=verify;m.evaluate=prior.evaluate
    m.label_path=lambda c:base.BASE/'raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train6.json')
    m.score();new=load(OUT/'analysis.json');old=load(previous.OUT/'analysis.json');lookup={(r['case'],r['model'],r['nominal_h']):r for r in old['table']};count=0
    for r in new['table']:
        if r['model'] in previous.MODELS:assert r==lookup[r['case'],r['model'],r['nominal_h']];count+=1
        if r['model'] in ADDED and r['nominal_h']==.4 and r['case']!='negative5':print({k:r[k] for k in ['case','model','mean_fde_px','joint_success']},flush=True)
    save(OUT/'score_checks.json',dict(previous26_rows_exact=count,total_rows=len(new['table'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','infer','score']);globals()[p.parse_args().action]()
