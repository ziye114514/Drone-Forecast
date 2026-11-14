"""Causal guard ablation, preserving all23 stage69 branches exactly."""
from pathlib import Path
import argparse,datetime,importlib.util,json,sys
from guard70 import *
import study69 as previous
base=previous.base
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage22'))
OUT=ROOT.parent/'output/stage70_guarded_event_force_20261021'
CASES=previous.CASES;load=base.load;save=base.save;sha=base.sha;rows=base.rows

def sources():return {p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),Path(__file__).with_name('guard70.py')]}
def freeze():
    previous.verify();assert not OUT.exists()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=CASES,
        source_hashes=sources(),input_hashes={c:sha(previous.OUT/c/'forecasts.jsonl') for c in CASES},
        label_hashes=load(previous.OUT/'protocol.json')['label_hashes'],parameters=PARAMS,controls=controls(),
        scope='Same known development recordings. No new labels or RAW inference. '
              'Guard sees only matured current-detector errors, not score labels. '
              'Success requires event advantage beyond constants and history, '
              'not merely returning history when all physical experts fail.'))
    print('70 guard frozen',flush=True)
def verify():
    previous.verify();p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    for c,h in p['input_hashes'].items():assert h==sha(previous.OUT/c/'forecasts.jsonl')
    return p
def infer():
    verify();sys.addaudithook(base.guard)
    for case in CASES:
        rr=rows(previous.OUT/case/'forecasts.jsonl');current=rows(base.REFERENCE/f'off_{case}/observations_forecasts.jsonl');g=Guard()
        folder=OUT/case;folder.mkdir(exist_ok=False);counts={m:{} for m in ADDED70}
        with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as f:
            for r,c in zip(rr,current):
                q=r['frontends']['event_force'];new=g.step(q,c['observation'])
                for m in PREVIOUS_MODELS:assert new['coefficients'].get(m)==q['coefficients'].get(m)
                for m,d in new['force_guard_diagnostic']['choices'].items():counts[m][d['chosen']]=counts[m].get(d['chosen'],0)+1
                f.write(json.dumps(dict(t=r['t'],frontends=dict(event_force=new)),allow_nan=False)+'\n')
        save(folder/'completion.json',dict(frames=len(rr),all23_previous_exact=True,labels_read=False,choices=counts));print(case,counts,flush=True)
def score():
    verify()
    spec=importlib.util.spec_from_file_location('score70_original22',ROOT/'research/stage22/study22.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=OUT;m.INPUT=OUT;m.CASES=list(CASES);m.FRONTENDS=['event_force'];m.MODELS=MODELS;m.verify=verify;m.evaluate=prior.evaluate
    m.label_path=lambda c:base.BASE/'raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train6.json')
    m.score();new=load(OUT/'analysis.json');old=load(previous.OUT/'analysis.json');lookup={(r['case'],r['model'],r['nominal_h']):r for r in old['table']};count=0
    for r in new['table']:
        if r['model'] in PREVIOUS_MODELS:assert r==lookup[r['case'],r['model'],r['nominal_h']];count+=1
        if r['model'] in ADDED70 and r['nominal_h']==.4 and r['case']!='negative5':print({k:r[k] for k in ['case','model','mean_fde_px','joint_success']},flush=True)
    save(OUT/'score_checks.json',dict(previous23_score_rows_exact=count,total_rows=len(new['table'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','infer','score']);globals()[p.parse_args().action]()
