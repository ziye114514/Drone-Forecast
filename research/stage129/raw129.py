"""Small actualRAW guard pilot; full8s evaluation remains cached diagnostic."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage129'),str(ROOT/'research/stage128')]
import run_FOV_guard_forecast as entry
from study129 import OUT,rows,sha,save,stable
from study128 import OUT as PREVIOUS
CASES=dict(prefix_natural6=dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=1.2,reference='natural6'),negative5=dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=.6,reference='negative5'))
def sources():
    ns=entry.build();h=ns['fingerprints']();ns.clear();h[str(Path(__file__).relative_to(ROOT))]=sha(__file__);return h
def freeze():
    assert not (OUT/'protocol_raw129.json').exists();save(OUT/'protocol_raw129.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),cases=CASES,
        scope='Only60 positive+30 negative actualRAW pilot, not a new8s end-to-end efficacy run. Check cached129 actual guard states and original71 equality128. First60 includes the invalid earlier FOV startup and guarded later lock.'))
def check(case):
    protocol=json.loads((OUT/'protocol_raw129.json').read_text('utf-8'));assert sources()==protocol['sources'];c=CASES[case];rr=rows(OUT/'raw'/case/'observations_forecasts.jsonl');old=rows(PREVIOUS/'raw'/c['reference']/'observations_forecasts.jsonl')[:len(rr)];cached=json.loads((OUT/(c['reference']+'.json')).read_text('utf-8'))[:len(rr)];names=set()
    for r,o,h in zip(rr,old,cached):
        q=r['frontends']['delayed_feedback'];p=o['frontends']['delayed_feedback'];w=q['learned_structure118'];prior=p['learned_structure118'];safe=w['support129']
        assert r['body_observation']==o['body_observation']
        for field in ['coefficients','predictions']:
            for name,value in p[field].items():assert q[field][name]==value,(r['t'],field,name);names.add(name)
        stripped={k:v for k,v in w.items() if k!='support129'};stripped['coefficients']={k:v for k,v in stripped['coefficients'].items() if k!='support_FOV_guard_kf129'};assert stable(stripped)==stable(prior)
        assert stable(safe)==stable(h['output']),(r['t'],'guard differs cached129')
        assert w['support126']['max_event_t'] is None or w['support126']['max_event_t']<round(r['t']*1e6)
    assert len(rr)==round(c['duration']*50)
    if case=='negative5':assert all(not r['frontends']['delayed_feedback']['valid'] for r in rr)
    else:assert len(names)==71
    completion=json.loads((OUT/'raw'/case/'completion.json').read_text('utf-8'));assert completion['source_hashes_unchanged'] and not completion['labels_read']
    save(OUT/'raw'/case/'checks129.json',dict(case=case,frames=len(rr),actual_RAW=True,original71_exact128=case!='negative5',old_worker_exact128=True,new_guard_exact129=True,cache_or_truth_read=False,all_cutoffs_valid=True,default_promoted=False))
    print(json.dumps(dict(case=case,frames=len(rr),checked=True)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','check']);p.add_argument('--case',choices=CASES);a=p.parse_args();freeze() if a.action=='freeze' else check(a.case)
