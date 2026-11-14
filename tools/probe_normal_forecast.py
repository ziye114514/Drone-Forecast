"""Same saved inputs; only integrate future headless-normal evolution."""
from pathlib import Path
import sys,json,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from normal_turn_forecast import NormalTurnForecast
from compare_final_pipeline import score,reference_at

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);results={}
    base=ROOT.parent/'output'
    specs=[('side',base/'shared_template_budget_20261027/side_visibility_final8s','dense_labels_side_110_1195'),
           ('train2',base/'rotor_force_30px_20261028/RAW_train2_allpixels6s','dense_labels_train2_40_46'),
           ('clear',base/'rotor_force_30px_20261028/RAW_clear2s',None)]
    for name,path,label in specs:
        rows=[json.loads(s) for s in (path/'predictions.jsonl').read_text('utf-8').splitlines()]
        observer=NormalTurnForecast();changed=[];reasons=collections.Counter();deltas=[]
        for r in rows:
            trend=observer.observe(r['t'],r['structure'],r['coefficient'],r.get('image_motion'))
            p,d=observer.correct(r['physical'],r['force'],trend);reasons[d['reason']]+=1
            if d['applied']:deltas.append(float(np.linalg.norm(d['corrections_px']['0.4'])))
            changed.append(dict(r,physical=p,normal_turn_forecast=d))
        # Freeze before opening manual targets.
        (out/(name+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in changed)+'\n','utf-8')
        result=dict(source_frames=len(rows),applied=len(deltas),reasons=dict(reasons),
            correction04_px_mean=None if not deltas else float(np.mean(deltas)),scores=None,RAW_retimed=False)
        if label:
            truth=sorted(json.loads((base/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
            result['scores']={k:score(v,truth) for k,v in [('original',rows),('normal_turn',changed)]}
            for v in result['scores'].values():v.pop('anchors',None)
        results[name]=result
    (out/'summary.json').write_text(json.dumps(results,indent=2),'utf-8')
    print(json.dumps({name:dict(applied=x['applied'],reasons=x['reasons'],
        scores=None if not x['scores'] else {k:v['physical_future_error_px']['0.4'] for k,v in x['scores'].items()}) for name,x in results.items()}))
if __name__=='__main__':main(sys.argv[1])
