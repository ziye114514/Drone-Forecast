"""Fixed fast-state/robust-rollout optimization, compact existing comparisons."""
from pathlib import Path
import argparse,datetime,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage132'),str(ROOT/'research/stage130')]
from fast132 import Fast,PARAMS,forecast
from optimize130 import CASES,BODY,INPUT,REFERENCE,rows,save,sha
OUT=ROOT.parent/'output/stage132_fast_robust_prediction_20261024'
def freeze():
    assert not OUT.exists();save(OUT/'protocol132.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources={p.name:sha(p) for p in [Path(__file__),ROOT/'research/stage132/fast132.py',ROOT/'research/stage130/motion130.py',ROOT/'research/stage23/robust23.py']},parameters=PARAMS,
        design='Reuse actual robust23 acceleration in faster107 state; no uncertain acceleration injected into velocity state. Two controls differ only by relative rotor-geometry velocity.131 damping internal-state approach failed; fixed before132 outcomes. Same exposed development pairs.'))
def run():
    protocol=json.loads((OUT/'protocol132.json').read_text('utf-8'));assert protocol['sources']=={p.name:sha(p) for p in [Path(__file__),ROOT/'research/stage132/fast132.py',ROOT/'research/stage130/motion130.py',ROOT/'research/stage23/robust23.py']};lookup={};summary=[]
    for case in CASES:
        bb=rows(BODY/case/'records.jsonl');gg=json.loads((INPUT/(case+'.json')).read_text('utf-8'));model=Fast();rr=[];cost=[]
        for b,g in zip(bb,gg):
            tick=time.perf_counter();o=model.step(b['t'],b['body'],g['output']['independent_models']);cost.append((time.perf_counter()-tick)*1000);rr.append(dict(t=b['t'],output=o))
        save(OUT/(case+'.json'),rr);lookup[case]={round(r['t'],8):r['output'] for r in rr};summary.append(dict(case=case,frames=len(rr),median_predictor_ms=float(np.median(cost))))
    old=json.loads(REFERENCE.read_text('utf-8'));pairs=[];table=[]
    for p in old['pairs']:
        for name in ['fast_body_acc','fast_geometry_acc']:
            c=lookup[p['case']][round(p['origin'],8)][name]['coefficient'];assert c is not None;prediction=forecast(c,p['actual_h']);pairs.append(dict(case=p['case'],h=p['h'],method=name,origin=p['origin'],target=p['target'],prediction=prediction,error_px=float(np.linalg.norm(np.array(prediction)-p['truth_bbox_center']))))
    for base in old['table']:
        if base['method']!='count_model':continue
        for name in ['fast_body_acc','fast_geometry_acc']:
            e=[p['error_px'] for p in pairs if p['case']==base['case'] and p['h']==base['h'] and p['method']==name];table.append(dict(base,method=name,mean_px=float(np.mean(e)) if e else None,median_px=float(np.median(e)) if e else None,rmse_px=float(np.sqrt(np.mean(np.square(e)))) if e else None))
    save(OUT/'optimization132.json',dict(availability=summary,table=table,pairs=pairs,changed_predictor=True,no_NEW_RAW_run_yet=True));print(json.dumps(dict(availability=summary,score04=[r for r in table if r['h']==.4])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();globals()[a.action]()
