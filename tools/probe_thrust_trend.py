"""Freeze experimental relative amplitude rollout before opening dev labels."""
from pathlib import Path
import json,sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from thrust_trend_forecast import ThrustTrendForecast
from compare_final_pipeline import score

def main(output):
    out=Path(output);out.mkdir(exist_ok=False,parents=True);base=ROOT.parent/'output';results={}
    specs=[('side','motion_consistency_30px_20261028/RAW_side_curved8s','dense_labels_side_110_1195'),
        ('train2','motion_consistency_30px_20261028/RAW_train2_curved6s','dense_labels_train2_40_46')]
    for name,folder,label in specs:
        rows=[json.loads(s) for s in (base/folder/'predictions.jsonl').read_text('utf-8').splitlines()]
        observer=ThrustTrendForecast();values=[]
        for r in rows:
            trend=observer.observe(r['t'],r['period_selection'],r['period'],r.get('axis_amplitude_bridge'))
            p,d=observer.correct(r['physical'],r['force'],r.get('normal_turn_forecast'),trend)
            values.append(dict(r,physical=p,thrust_trend_forecast=d))
        (out/(name+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in values)+'\n','utf-8')
        truth=sorted(json.loads((base/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        results[name]=dict(scores={k:score(v,truth)['physical_future_error_px'] for k,v in [('current',rows),('amplitude_trend',values)]},
            reasons=dict(collections.Counter(r['thrust_trend_forecast']['reason'] for r in values)),
            same_current_observations_and_slots=True,RAW_FPS_measured=False)
    (out/'summary.json').write_text(json.dumps(results,indent=2),'utf-8');print(json.dumps(results))
if __name__=='__main__':main(sys.argv[1])
