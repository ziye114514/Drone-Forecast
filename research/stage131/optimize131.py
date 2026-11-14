"""Fixed damped-motion optimization; no new geometric checks or RAW controls."""
from pathlib import Path
import argparse,datetime,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage131'),str(ROOT/'research/stage130')]
from dynamics131 import Optimized131,PARAMS,forecast
from optimize130 import CASES,BODY,INPUT,REFERENCE,rows,save,sha,stable
OUT=ROOT.parent/'output/stage131_damped_prediction_20261024'
def freeze():
    assert not OUT.exists();save(OUT/'protocol131.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),parameters=PARAMS,sources={p.name:sha(p) for p in [Path(__file__),ROOT/'research/stage131/dynamics131.py',ROOT/'research/stage130/motion130.py']},
        design='Two fixed131 alternatives: body-only damped acceleration vs same model with current geometrical-center relative velocity. Same guarded observations and old128 endpoint pairs.130 outcomes known; development optimization, not blind.'))
def run():
    p=json.loads((OUT/'protocol131.json').read_text('utf-8'));assert p['sources']=={q.name:sha(q) for q in [Path(__file__),ROOT/'research/stage131/dynamics131.py',ROOT/'research/stage130/motion130.py']};summary=[];lookup={}
    for case in CASES:
        bb=rows(BODY/case/'records.jsonl');gg=json.loads((INPUT/(case+'.json')).read_text('utf-8'));model=Optimized131();result=[];cost=[]
        for b,g in zip(bb,gg):
            tick=time.perf_counter();o=model.step(b['t'],b['body'],g['output']['independent_models']);cost.append((time.perf_counter()-tick)*1000);result.append(dict(t=b['t'],output=o))
        save(OUT/(case+'.json'),result);lookup[case]={round(r['t'],8):r['output'] for r in result};summary.append(dict(case=case,frames=len(result),median_predictor_ms=float(np.median(cost)),geometry_velocity_updates=sum(r['output']['damped_geometry']['source'] is not None for r in result)))
    old=json.loads(REFERENCE.read_text('utf-8'));pairs=[];table=[]
    for p in old['pairs']:
        for name,q in lookup[p['case']][round(p['origin'],8)].items():
            pred=forecast(q['coefficient'],p['actual_h']);assert pred is not None;pairs.append(dict(case=p['case'],h=p['h'],method=name,target=p['target'],origin=p['origin'],prediction=pred,error_px=float(np.linalg.norm(np.array(pred)-p['truth_bbox_center']))))
    for base in old['table']:
        if base['method']!='count_model':continue
        for name in ['damped_body','damped_geometry']:
            e=[p['error_px'] for p in pairs if p['case']==base['case'] and p['h']==base['h'] and p['method']==name];assert len(e)==base['paired'];table.append(dict(base,method=name,mean_px=float(np.mean(e)) if e else None,median_px=float(np.median(e)) if e else None,rmse_px=float(np.sqrt(np.mean(np.square(e)))) if e else None))
    save(OUT/'optimization131.json',dict(availability=summary,table=table,pairs=pairs,changed_predictor=True,added_damped_acceleration=True,no_NEW_RAW_run_yet=True))
    print(json.dumps(dict(availability=summary,score04=[r for r in table if r['h']==.4])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();globals()[a.action]()
