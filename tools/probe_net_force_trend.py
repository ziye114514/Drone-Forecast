"""One fixed model hypothesis, frozen observations; no horizon/parameter sweep."""
from pathlib import Path
import sys,json,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from net_force_trend import correct,jerk_kernel
from compare_final_pipeline import reference_at
from score_fred_adaptation import matched


if __name__=='__main__':
    base=ROOT.parent/'output';out=base/'target25_continuity_20261031';result={}
    for name,folder,label in [('other','RAW_other_center6s','dense_labels_train2_40_46'),('side','../reacquisition_motion_20261029/RAW_side_default8s','dense_labels_side_110_1195')]:
        rows=[json.loads(x) for x in (out/folder/'predictions.jsonl').read_text().splitlines()];new=[];applied=[]
        for r in rows:
            p,d=correct(r['physical'],r['force'],r['image_motion'],r.get('adaptive_force_memory'))
            new.append(dict(r,physical=p,net_force_trend=d))
            if d['applied']:applied.append(r['t'])
        labels=sorted(json.loads((base/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        result[name]=dict(matched=matched(rows,new,lambda t:reference_at(labels,t)),applied_times=applied)
        (out/(name+'_net_trend.jsonl')).write_text('\n'.join(json.dumps(r,allow_nan=False) for r in new)+'\n','utf-8')
    # Numerical integral check independent of the dataset and annotations.
    x=np.linspace(0,.4,10001);expected=np.trapezoid((.4-x)*x*np.exp(-x/.35),x)
    assert abs(jerk_kernel(-1/.35,.4)-expected)<1e-10
    (out/'net_force_trend_scores.json').write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({k:dict(matched={a:b for a,b in v['matched'].items() if a!='records'},applied=len(v['applied_times'])) for k,v in result.items()},indent=2))
