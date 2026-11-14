"""One fixed fallback change, compare exposed development observations."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent;sys.path[:0]=[str(ROOT/'research/stage139')]
from dense139 import DenseGate,PARAMS,forecast
OUT=WORK/'output/stage139_dense_fallback_20261024';OLD=WORK/'output/stage135_optimized_forecast_20261024';BODY=WORK/'output/stage124_occupancy_support_20261023'
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8')
def main():
    assert not OUT.exists();save(OUT/'protocol139.json',dict(parameters=PARAMS,source_sha256=hashlib.sha256((ROOT/'research/stage139/dense139.py').read_bytes()).hexdigest(),type='Fixed-input development optimization; no RAW deployment yet.'))
    lookup={};availability=[]
    for case in ['extension1','extension2','natural6','continuous6','negative5']:
        bb=[json.loads(s) for s in (BODY/case/'records.jsonl').read_text('utf-8').splitlines()];cc=load(OLD/(case+'.json'));gate=DenseGate();rr=[]
        for b,c in zip(bb,cc):rr.append(dict(t=b['t'],output=gate.step(b['t'],b['body'],c['output'])))
        save(OUT/(case+'.json'),rr);lookup[case]={round(r['t'],8):r['output'] for r in rr};availability.append(dict(case=case,frames=len(rr),fallback_active=sum(r['output']['dense_motion_fallback']['active'] for r in rr)))
    # Outputs are complete before historical manual endpoint references are read.
    reference=load(WORK/'output/stage128_RAW_independent_support_20261023/scores128.json');pairs=[];table=[]
    for p in reference['pairs']:
        c=lookup[p['case']][round(p['origin'],8)]['coefficients']['optimized_geometry'];pred=forecast(c,p['actual_h']);pairs.append(dict(case=p['case'],h=p['h'],origin=p['origin'],target=p['target'],error_px=float(np.linalg.norm(np.array(pred)-p['truth_bbox_center']))))
    previous=load(OLD/'optimization135.json')
    for base in previous['table']:
        if base['method']!='optimized_geometry':continue
        e=[p['error_px'] for p in pairs if p['case']==base['case'] and p['h']==base['h']];assert len(e)==base['paired'];after=float(np.mean(e)) if e else None;table.append(dict(case=base['case'],h=base['h'],paired=len(e),before_px=base['mean_px'],after_px=after))
    result=dict(availability=availability,table=table,pairs=pairs,development_only=True);save(OUT/'scores139.json',result);print(json.dumps(dict(availability=availability,score04=[r for r in table if r['h']==.4])),flush=True)
if __name__=='__main__':main()
