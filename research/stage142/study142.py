from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent;sys.path[:0]=[str(ROOT/'research/stage142'),str(ROOT/'research/stage135')]
from feature_model142 import Model,PARAMS
from predictor135 import forecast
OUT=WORK/'output/stage142_frequency_model_pilot_20261024'
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8')
def main():
    assert not OUT.exists();save(OUT/'protocol142.json',dict(parameters=PARAMS,sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'research/stage142/feature_model142.py',ROOT/'src/event_drone_forecast/paper_physics.py']},type='Current141 observations, no new RAW forecast yet; exposed development clip.'))
    rows=[json.loads(s) for s in (WORK/'output/stage141_periodic_axis_20261024/continuous6/predictions.jsonl').read_text('utf-8').splitlines()];m=Model();rr=[m.step(r) for r in rows];save(OUT/'forecasts142.json',rr)
    prefix=Model()
    for r,ref in zip(rows[:100],rr[:100]):assert prefix.step(r)==ref
    lookup={round(r['t'],8):r for r in rr};old=load(WORK/'output/stage128_RAW_independent_support_20261023/scores128.json');before=load(WORK/'output/stage140_visible_fallback_20261024/scores139.json');pairs=[];table=[]
    for p in old['pairs']:
        if p['case']!='continuous6':continue
        r=lookup[round(p['origin'],8)];pred=forecast(r['coefficient'],p['actual_h']);pairs.append(dict(h=p['h'],origin=p['origin'],error_px=float(np.linalg.norm(np.array(pred)-p['truth_bbox_center'])),active=r['active']))
    for h in [.1,.2,.4]:
        values=[p for p in pairs if p['h']==h];b=next(r for r in before['table'] if r['case']=='continuous6' and r['h']==h);table.append(dict(h=h,pairs=len(values),before_px=b['after_px'],after_px=float(np.mean([r['error_px'] for r in values])),active_origins=sum(r['active'] for r in values)))
    result=dict(table=table,pairs=pairs,frames=len(rr),supported=sum(r['supported'] for r in rr),active=sum(r['active'] for r in rr),maximum_updates=max(r['updates'] for r in rr),prefix_exact=True,development_only=True);save(OUT/'scores142.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
