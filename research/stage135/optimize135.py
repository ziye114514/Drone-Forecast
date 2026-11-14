"""Compact fixed-input optimization comparison, with necessary prefix/empty."""
from pathlib import Path
import argparse,datetime,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage135'),str(ROOT/'research/stage130')]
from predictor135 import Predictor,PARAMS,forecast
from optimize130 import CASES,BODY,INPUT,REFERENCE,rows,save,sha,stable
OUT=ROOT.parent/'output/stage135_optimized_forecast_20261024'
def sources():return {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'research/stage135/predictor135.py',ROOT/'research/stage132/fast132.py',ROOT/'research/stage130/motion130.py',ROOT/'research/stage23/robust23.py']}
def freeze():
    assert not OUT.exists();save(OUT/'protocol135.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),parameters=PARAMS,
        design='One deployed predictor change + matched body-only velocity control. Reuse qualified-model memory as target support; no learned geometry means CV. No ROI/conic/lock threshold change. Old development outcomes known; same endpoint pairs.'))
def run():
    p=json.loads((OUT/'protocol135.json').read_text('utf-8'));assert p['sources']==sources();lookup={};summary=[]
    for case in CASES:
        bb=rows(BODY/case/'records.jsonl');gg=json.loads((INPUT/(case+'.json')).read_text('utf-8'));m=Predictor();rr=[];cost=[]
        for b,g in zip(bb,gg):
            tick=time.perf_counter();o=m.step(b['t'],b['body'],g['output']['independent_models']);cost.append((time.perf_counter()-tick)*1000);rr.append(dict(t=b['t'],output=o))
        save(OUT/(case+'.json'),rr);lookup[case]={round(r['t'],8):r['output'] for r in rr};prefix=Predictor()
        for b,g,r in zip(bb[:60],gg[:60],rr[:60]):assert stable(prefix.step(b['t'],b['body'],g['output']['independent_models']))==stable(r['output'])
        summary.append(dict(case=case,frames=len(rr),median_predictor_ms=float(np.median(cost)),prefix_exact=True,structure_supported=sum(r['output']['structure_supported'] for r in rr)))
    old=json.loads(REFERENCE.read_text('utf-8'));pairs=[];table=[]
    for p in old['pairs']:
        for name,c in lookup[p['case']][round(p['origin'],8)]['coefficients'].items():
            assert c is not None;prediction=forecast(c,p['actual_h']);pairs.append(dict(case=p['case'],h=p['h'],method=name,origin=p['origin'],target=p['target'],prediction=prediction,error_px=float(np.linalg.norm(np.array(prediction)-p['truth_bbox_center']))))
    for base in old['table']:
        if base['method']!='count_model':continue
        for name in ['optimized_geometry','matched_body']:
            e=[p['error_px'] for p in pairs if p['case']==base['case'] and p['h']==base['h'] and p['method']==name];assert len(e)==base['paired'];table.append(dict(base,method=name,mean_px=float(np.mean(e)) if e else None,median_px=float(np.median(e)) if e else None,rmse_px=float(np.sqrt(np.mean(np.square(e)))) if e else None))
    save(OUT/'optimization135.json',dict(availability=summary,table=table,pairs=pairs,changed_predictor=True,no_NEW_RAW_run_yet=True));print(json.dumps(dict(availability=summary,score04=[r for r in table if r['h']==.4])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();globals()[a.action]()
