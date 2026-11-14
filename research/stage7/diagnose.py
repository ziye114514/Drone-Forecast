"""Posthoc observation quality audit; no labels enter automatic estimation."""
import json,collections
import numpy as np
from study import OUT,FRONTS,load,save,read_labels

def main():
    rows=[]
    for i in range(11):
        folder=OUT/'runs'/f'sequence{i}';meta=load(folder/'completion.json')
        obs=[json.loads(x) for x in (folder/'observations.jsonl').read_text().splitlines()];times=np.array([r['t'] for r in obs])
        for front in FRONTS:
            seen=[];last=None
            for r in obs:
                if r[front]['detected']:last=(r['t'],r[front])
                seen.append(last)
            for label in read_labels(i):
                k=int(np.searchsorted(times,label['t']+1e-10,side='right')-1)
                if k<0:continue
                recent=seen[k];fresh=recent is not None and label['t']-recent[0]<=.05+1e-10
                b=np.array(label['bbox']);diagonal=float(np.linalg.norm(b[2:]-b[:2]));err=None
                if fresh:err=float(np.linalg.norm(np.array(recent[1]['center'])-label['center']))
                r=obs[k]['rotor']
                rows.append(dict(sequence=i,role=meta['role'],front=front,t=label['t'],diagonal_px=diagonal,
                    size_band='<50px' if diagonal<50 else '50-150px' if diagonal<150 else '>=150px',
                    fresh=fresh,center_error_px=err,error_over_diagonal=err/diagonal if err is not None else None,
                    four_current=int(r.get('observed_rotor_count',0))==4,
                    frequency_quality=float(r.get('frequency_quality',0.)),geometry_quality=float(r.get('geometry_quality',0.))))
    groups={'all_local':list(range(11)),'additional_train_transfer':[0,4,5,6,7,10],'official_test_subset':[8,9]}
    results=[]
    for group,ids in groups.items():
        for front in FRONTS:
            for size in ['all','<50px','50-150px','>=150px']:
                selected=[r for r in rows if r['sequence'] in ids and r['front']==front and (size=='all' or r['size_band']==size)]
                measured=[r for r in selected if r['fresh']];errors=[r['center_error_px'] for r in measured]
                results.append(dict(group=group,front=front,size_band=size,label_times=len(selected),fresh=len(measured),
                    fresh_coverage=len(measured)/max(1,len(selected)),mean_center_error_px=float(np.mean(errors)) if errors else None,
                    median_center_error_px=float(np.median(errors)) if errors else None,
                    mean_normalized_error=float(np.mean([r['error_over_diagonal'] for r in measured])) if measured else None,
                    fraction_fresh_and_within_quarter_diagonal=sum(r['error_over_diagonal']<=.25 for r in measured)/max(1,len(selected)),
                    four_current_times=sum(r['four_current'] for r in selected),frequency_quality_times=sum(r['frequency_quality']>=.35 for r in selected)))
    save(OUT/'observation_budget.json',dict(results=results,details=rows,
        caution='Past50ms latest measurement compared to current label; residual includes latency and target motion. No independent RPM/pose truth.'))
    print(json.dumps([r for r in results if r['size_band']=='all'],ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
