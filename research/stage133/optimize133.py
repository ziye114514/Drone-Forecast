"""Run the final concrete predictor optimization on existing fixed inputs."""
from pathlib import Path
import argparse,collections,datetime,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage133'),str(ROOT/'research/stage130')]
from predictor133 import Predictor,PARAMS,forecast
from optimize130 import CASES,BODY,INPUT,REFERENCE,rows,save,sha,stable
OUT=ROOT.parent/'output/stage133_adaptive_prediction_20261024'
def sources():return {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'research/stage133/predictor133.py',ROOT/'research/stage132/fast132.py',ROOT/'research/stage130/motion130.py',ROOT/'research/stage23/robust23.py',ROOT/'research/stage70/guard70.py']}
def freeze():
    assert not OUT.exists();save(OUT/'protocol133.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),parameters=PARAMS,
        design='Actual predictor changed, not observation gate. Reset acceleration history on existing geometry gap40ms, reuse70 delayed error guard instead of reinventing.130/131/132 outcomes known; fixed before133 outcomes. Same old development endpoint pairs; no claim of independent efficacy. Minimal RAW version next, not old72 controls.'))
def run():
    p=json.loads((OUT/'protocol133.json').read_text('utf-8'));assert p['sources']==sources();lookup={};summary=[]
    for case in CASES:
        bb=rows(BODY/case/'records.jsonl');gg=json.loads((INPUT/(case+'.json')).read_text('utf-8'));model=Predictor();rr=[];cost=[]
        for b,g in zip(bb,gg):
            tick=time.perf_counter();o=model.step(b['t'],b['body'],g['output']['independent_models']);cost.append((time.perf_counter()-tick)*1000);rr.append(dict(t=b['t'],output=o))
        save(OUT/(case+'.json'),rr);lookup[case]={round(r['t'],8):r['output'] for r in rr};prefix=Predictor()
        for b,g,o in zip(bb[:60],gg[:60],rr[:60]):assert stable(prefix.step(b['t'],b['body'],g['output']['independent_models']))==stable(o['output'])
        summary.append(dict(case=case,frames=len(rr),median_predictor_ms=float(np.median(cost)),choices=dict(collections.Counter(r['output']['chosen'] for r in rr if r['output']['valid'])),prefix_exact=True))
    old=json.loads(REFERENCE.read_text('utf-8'));pairs=[];table=[]
    for p in old['pairs']:
        q=lookup[p['case']][round(p['origin'],8)];c=q['coefficient']
        if c is None:continue
        prediction=forecast(c,p['actual_h']);pairs.append(dict(case=p['case'],h=p['h'],method='optimized133',origin=p['origin'],target=p['target'],prediction=prediction,error_px=float(np.linalg.norm(np.array(prediction)-p['truth_bbox_center'])),chosen=q['chosen']))
    for base in old['table']:
        if base['method']!='count_model':continue
        e=[p['error_px'] for p in pairs if p['case']==base['case'] and p['h']==base['h']];table.append(dict(base,method='optimized133',paired=len(e),mean_px=float(np.mean(e)) if e else None,median_px=float(np.median(e)) if e else None,rmse_px=float(np.sqrt(np.mean(np.square(e)))) if e else None))
    save(OUT/'optimization133.json',dict(availability=summary,table=table,pairs=pairs,changed_predictor=True,no_NEW_RAW_run_yet=True));print(json.dumps(dict(availability=summary,score04=[r for r in table if r['h']==.4])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();globals()[a.action]()
