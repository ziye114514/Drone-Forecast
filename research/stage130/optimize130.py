"""Short development comparison of two actual predictor changes."""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage130'),str(ROOT/'research/stage124')]
from motion130 import Optimized,PARAMS
from study124 import CASES,OUT as BODY,rows,save,sha,stable
INPUT=ROOT.parent/'output/stage129_FOV_guard_20261024';REFERENCE=ROOT.parent/'output/stage128_RAW_independent_support_20261023/scores128.json'
OUT=ROOT.parent/'output/stage130_prediction_optimization_20261024'
def freeze():
    assert not OUT.exists();save(OUT/'protocol130.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),parameters=PARAMS,sources={p.name:sha(p) for p in [Path(__file__),ROOT/'research/stage130/motion130.py']},
        cases=list(CASES),design='Two implementations fixed before measuring130 outputs. All datasets/old outcomes already exposed; developmental optimization, not independent generalization test. No parameter sweep. Compare same target/origin pairs as128.',
        scope='Current129 guarded model observations+same-time body, no geometric-gate change, no absolute hidden-truth anchor. Predictor-only latency measured separately.'))
def verify():
    p=json.loads((OUT/'protocol130.json').read_text('utf-8'));assert p['sources']=={q.name:sha(q) for q in [Path(__file__),ROOT/'research/stage130/motion130.py']};return p
def run():
    verify();summary=[]
    for case in CASES:
        bb=rows(BODY/case/'records.jsonl');gg=json.loads((INPUT/(case+'.json')).read_text('utf-8'));model=Optimized();out=[]
        import time
        costs=[]
        for b,g in zip(bb,gg):
            assert b['t']==g['t'];start=time.perf_counter();o=model.step(b['t'],b['body'],g['output']['independent_models']);costs.append((time.perf_counter()-start)*1000);out.append(dict(t=b['t'],output=o))
        save(OUT/(case+'.json'),out)
        prefix=Optimized()
        for b,g,o in zip(bb[:60],gg[:60],out[:60]):assert stable(prefix.step(b['t'],b['body'],g['output']['independent_models']))==stable(o['output'])
        summary.append(dict(case=case,frames=len(out),median_predictor_ms=float(np.median(costs)),motion_updates=sum(o['output']['relative_velocity']['source'] is not None for o in out),offset_updates=sum(o['output']['adaptive_offset']['source'] is not None for o in out),prefix_exact=True))
    reference=json.loads(REFERENCE.read_text('utf-8'));pairs=[];table=[];look={case:{round(r['t'],8):r['output'] for r in json.loads((OUT/(case+'.json')).read_text('utf-8'))} for case in CASES}
    for p in reference['pairs']:
        for name,cfg in look[p['case']][round(p['origin'],8)].items():
            c=cfg['coefficient'];assert c is not None;pred=np.array(c['position'])+p['actual_h']*np.array(c['velocity']);error=float(np.linalg.norm(pred-p['truth_bbox_center']));pairs.append(dict(case=p['case'],h=p['h'],method=name,origin=p['origin'],target=p['target'],prediction=pred.tolist(),error_px=error))
    for base in reference['table']:
        if base['method']!='count_model':continue
        for name in ['adaptive_offset','relative_velocity']:
            e=[p['error_px'] for p in pairs if p['case']==base['case'] and p['h']==base['h'] and p['method']==name];assert len(e)==base['paired']
            table.append(dict(base,method=name,mean_px=float(np.mean(e)) if e else None,median_px=float(np.median(e)) if e else None,rmse_px=float(np.sqrt(np.mean(np.square(e)))) if e else None))
    save(OUT/'optimization130.json',dict(availability=summary,table=table,pairs=pairs,changed_predictor=True,same_guarded_observations=True,no_NEW_RAW_run_yet=True))
    print(json.dumps(dict(availability=summary,score04=[r for r in table if r['h']==.4])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();globals()[a.action]()
