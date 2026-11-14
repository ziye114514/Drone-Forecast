"""Frozen cached routing experiment, same scoring and no future feedback."""
from pathlib import Path
from types import FunctionType
import argparse,datetime,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage88'),str(ROOT/'research/stage90')]
from inventory88 import save,load,sha
import score88
from router92 import Router92
from paired88 import prior
OUT=ROOT.parent/'output/stage92_complementary_router_20261022'
BASE=ROOT.parent/'output/stage88_causal_event_controls_20261022'
KIN=ROOT.parent/'output/stage90_train_only_kinematic_20261022'
AUTO=ROOT.parent/'output/stage78_RAW_separated_forecast_20261021'
METHODS=['primary','cv','trusted_acc','kf_tuned','history_router','history_fixed','fallback_router','full_router','full_fixed']
CASES=['clear6','side6','train3','extra1','extra2','extra4']


def sources():return {str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('router92.py'),ROOT/'research/stage29/select29.py']}
def rows(p):return score88.rows(p)


def freeze():
    assert not OUT.exists()
    save(OUT/'protocol92.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        selected_sigma_sha256=sha(KIN/'selected_sigma.json'),reuse='Actual29 delayed exponential weights, delay.2s/expiry.26s/loss35px/rate.08/share.01, original numerical constants.',
        history_experts=['cv','trusted_acc','kf_tuned'],full_experts=['primary','cv','trusted_acc','kf_tuned'],
        initial='First expert .9, remaining mass equal: originalCV for history pool, original primary for full pool.',
        rule='fallback_router keeps primary when physical fresh, otherwise learned history mixture; full_router always mixes four candidates.',
        controls='Fixed initial-weight mixtures alongside adaptive mixtures. No thresholds/search/recording-specific choice.',
        feedback='New automatic detector position in self recordings, past provided position in FRED. No future reference targets.',
        scope='Cached forecast-layer upgrade only; not new RAW run or end-to-end realtime. Previous primary untouched.',
        default_promoted=False))
    print('92 complementary router protocol frozen',flush=True)


def verify():
    p=load(OUT/'protocol92.json');assert p['source_hashes']==sources();assert p['selected_sigma_sha256']==sha(KIN/'selected_sigma.json');return p


def run():
    verify();checks=[]
    for case in CASES+[f'fred{s}' for s in [0,4,6,10,8,9]]:
        if case.startswith('fred'):
            seq=int(case[4:]);a=rows(BASE/'fred'/f'sequence{seq}'/'forecasts.jsonl');b=rows(KIN/'fred'/f'sequence{seq}'/'forecasts.jsonl')
            obs=[]
            for r in a:
                audit=r['input_audit'];box=audit['bbox'];z=None if audit['current_label_t'] is None else ((np.array(box[:2])+box[2:])/2).tolist()
                obs.append(dict(detected=z is not None,center=z,confidence=1. if z is not None else 0.))
            dest=OUT/'fred'/f'sequence{seq}'
        else:
            a=rows(BASE/'self'/case/'forecasts.jsonl');b=load(KIN/'self'/case/'forecasts.json');obs=[r['observation'] for r in rows(AUTO/case/'observations_forecasts.jsonl')]
            dest=OUT/'self'/case
        assert len(a)==len(b)==len(obs);dest.mkdir(parents=True,exist_ok=False)
        model=Router92();result=[];cost=[]
        with (dest/'forecasts.jsonl').open('x',encoding='utf-8') as f:
            for p,k,z in zip(a,b,obs):
                assert p['t']==k['t'] and p['valid']==k['valid'];started=time.perf_counter();v=model.step(p,k,z);cost.append((time.perf_counter()-started)*1000)
                if v['valid']:assert v['coefficients']['primary']==p['coefficients']['event']
                f.write(json.dumps(v,allow_nan=False)+'\n');result.append(v)
        prefix=Router92();limit=min(100,len(a))
        short=[prefix.step(p,k,z) for p,k,z in zip(a[:limit],b[:limit],obs[:limit])];assert short==result[:limit]
        item=dict(case=case,ticks=len(result),actual_cache_prefix_exact=True,source_hashes_unchanged=True,
            router_median_ms=float(np.median(cost)),router_p95_ms=float(np.percentile(cost,95)),old_primary_preserved=True)
        save(dest/'completion.json',item);checks.append(item);print(case,'92 complete',flush=True)
    save(OUT/'checks92.json',dict(records=checks,all_prefixes_exact=True,old_primary_preserved=True,RAW_rerun=False,default_promoted=False))


def score():
    verify();assert not (OUT/'scores92.json').exists();load(OUT/'checks92.json')
    (OUT/'annotations').mkdir(exist_ok=True)
    for seq in [0,4,6,10,8,9]:(OUT/'annotations'/f'{seq}.txt').write_bytes((BASE/'annotations'/f'{seq}.txt').read_bytes())
    scoring=FunctionType(score88.fred_score.__code__,dict(score88.fred_score.__globals__,OUT=OUT,METHODS=METHODS))
    # Completion's case field would duplicate scorer's case, so use isolated load.
    def score_load(path):
        v=load(path)
        if str(path).endswith('completion.json'):v={k:x for k,x in v.items() if k!='case'}
        return v
    scoring=FunctionType(scoring.__code__,dict(scoring.__globals__,load=score_load))
    fred,audit=scoring(load(BASE/'protocol88.json'));self_detail=[]
    original=load(AUTO/'scoring/analysis.json')['details']
    for case in CASES:
        pred={round(r['t'],8):r for r in rows(OUT/'self'/case/'forecasts.jsonl')}
        for r in original:
            if r['case']!=case or r['frontend']!='event_force' or r['model']!='mmae_state_fresh' or r['nominal_h']!=.4 or r['target_state']!='present':continue
            v=pred.get(round(r.get('anchor_t',-100),8));valid=bool(v and v['valid'])
            for name in METHODS:
                q=dict(case=case,method=name,horizon=.4,anchor_t=r.get('anchor_t'),valid=valid)
                if valid:q['fde']=float(np.linalg.norm(np.array(prior.evaluate(v['coefficients'][name],r['actual_h']))-r['target_center']))
                if valid and name=='primary':assert abs(q['fde']-r['error_px'])<1e-9
                self_detail.append(q)
    table=score88.summarise(fred+self_detail)
    test={m:float(np.mean([r['fde'] for r in fred if r['case'] in ['fred8','fred9'] and r['method']==m and r['horizon']==.4 and r['valid']])) for m in METHODS}
    save(OUT/'endpoint_details92.json',dict(fred=fred,self=self_detail))
    save(OUT/'scores92.json',dict(table=table,test_04_mean_fde=test,old_primary_scores_exact78=True,
        default_promoted=False,no_new_independent_RAW_run=True,no_RPM_gain_claim=True))
    print('92 fixed router test',json.dumps(test),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','score']);a=p.parse_args();globals()[a.action]()
