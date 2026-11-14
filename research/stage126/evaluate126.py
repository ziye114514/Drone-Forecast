"""Post-inference sparse future-BOX scores, complete failure denominators."""
from pathlib import Path
import collections,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage124'),str(ROOT/'research/stage125'),str(ROOT/'research/stage126')]
from study124 import OUT as INPUT,CASES,rows,save
from study126 import OUT as RAW,CASES as RAWCASES
from hybrid125 import NAMES
OUT=ROOT.parent/'output/stage125_complementary_support_20261023'
METHODS=['body_KF','count_model','occupancy_model','hybrid_model']
def main():
    # No efficacy targets until all actualRAW jobs and active prefix finish.
    for c in RAWCASES:assert (RAW/'raw'/c/'checks126.json').exists(),c
    table=[];pairs=[];negative=[]
    for case,cfg in CASES.items():
        rr=rows(INPUT/case/'records.jsonl');hh=json.loads((OUT/(case+'.json')).read_text('utf-8'));assert len(rr)==len(hh)
        if case=='negative5':
            negative.append(dict(case=case,frames=len(rr),valid=sum(r['valid'] for r in rr),geometry_current=sum(h['output']['center']['current'] for h in hh)));continue
        data=json.loads((ROOT.parent/f'output/raw_only_project/evaluation_data/train{case[-1]}.json').read_text('utf-8-sig'))
        labels=[l for l in data['labels'] if cfg['start']<l['t']<=cfg['start']+cfg['duration'] and l['state']=='present' and l.get('bbox')]
        times=np.array([r['t'] for r in rr])
        for horizon in [.1,.2,.4]:
            eligible=[l for l in labels if l['t']>cfg['start']+horizon];errs={m:[] for m in METHODS};missing=collections.Counter()
            for l in eligible:
                i=int(np.searchsorted(times,l['t']-horizon+1e-10,side='right')-1)
                if i<0 or l['t']-horizon-times[i]>.02000001:missing['no_origin']+=1;continue
                r=rr[i]
                if not r['valid']:missing['not_published']+=1;continue
                coeff=dict(body_KF=r['outputs']['count_global']['forecast']['coefficients'].get(NAMES[0]),
                    count_model=r['outputs']['count_global']['forecast']['coefficients'].get(NAMES[2]),
                    occupancy_model=r['outputs']['occupancy_roi']['forecast']['coefficients'].get(NAMES[2]),hybrid_model=hh[i]['output']['forecast']['coefficients'].get(NAMES[2]))
                if not all(v is not None for v in coeff.values()):missing['method_uninitialized']+=1;continue
                truth=(np.array(l['bbox'][:2])+l['bbox'][2:])/2;dt=l['t']-r['t'];pred={}
                for m,c in coeff.items():
                    assert c['acceleration']==[0.,0.];p=np.array(c['position'])+dt*np.array(c['velocity']);err=float(np.linalg.norm(p-truth));errs[m].append(err);pred[m]=dict(position=p.tolist(),error_px=err)
                pairs.append(dict(case=case,h=horizon,target=l['t'],origin=r['t'],actual_h=dt,truth_bbox_center=truth.tolist(),predictions=pred))
            for m,e in errs.items():table.append(dict(case=case,h=horizon,method=m,eligible=len(eligible),paired=len(e),missing=dict(missing),mean_px=float(np.mean(e)) if e else None,
                median_px=float(np.median(e)) if e else None,rmse_px=float(np.sqrt(np.mean(np.square(e)))) if e else None))
    save(RAW/'scores126.json',dict(table=table,pairs=pairs,negative=negative,predictions_before_targets=True,
        caveat='Sparse manual event-image BOX centers, not COM. Previously exposed development data; no visibility/RPM/thrust truth. All methods use identical eligible origins.'))
    print(json.dumps([r for r in table if r['h']==.4]),flush=True)
if __name__=='__main__':main()
