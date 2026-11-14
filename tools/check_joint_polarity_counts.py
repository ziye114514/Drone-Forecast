"""Paired native count microbenchmark, with exact current planes checked."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from fast_structure_front import count_pair,roi_bounds,NativeROI
from live_decode import native_batches


def old(events,bounds,t,selector):
    x,y,X,Y=bounds;w,h=X-x,Y-y
    begin=int(np.searchsorted(events['t'],round(t*1e6)-20000));e=selector.select(events[begin:],bounds)
    index=(e['y'].astype(np.int32)-y)*w+(e['x'].astype(np.int32)-x)
    return [np.bincount(index[e['p']==p],minlength=w*h).reshape(h,w).astype(np.float32) for p in (1,0)]


def run(source,output,shared_roi=False):
    source=Path(source);out=Path(output);out.mkdir(exist_ok=False,parents=True)
    protocol=json.loads((source/'protocol.json').read_text('utf-8'))
    data=list(map(json.loads,(source/'predictions.jsonl').read_text('utf-8').splitlines()))
    names=['joint_full','joint_shared_roi'] if shared_roi else ['old','joint']
    selectors={n:NativeROI() for n in names};outer=NativeROI()
    cost={n:[] for n in selectors};compared=0;retained=[]
    for row,(t,events) in zip(data,native_batches(protocol['raw'],protocol['actual_iterator_start'],protocol['actual_duration'],False,protocol['native_step_us'])):
        assert abs(row['t']-t)<1e-8
        if not row['body'].get('detected'):continue
        bounds=roi_bounds(row['body']['bbox'])
        expanded=roi_bounds(row['body']['bbox'],padding=.35)
        assert all(bounds[:2]>=expanded[:2]) and all(bounds[2:]<=expanded[2:])
        selected=outer.select(events,expanded) if shared_roi else events
        retained.append(len(selected)/max(1,len(events)))
        for rep in range(3):
            values={}
            for name in (names if (compared+rep)%2==0 else names[::-1]):
                source_events=selected if name=='joint_shared_roi' else events
                a=time.perf_counter();values[name]=(old if name=='old' else count_pair)(source_events,bounds,t,selectors[name])
                cost[name].append((time.perf_counter()-a)*1000)
            assert all(np.array_equal(a,b) for a,b in zip(values[names[0]],values[names[1]]))
        compared+=1
    result=dict(native_ROIs=compared,paired_calls_per_method=len(cost[names[0]]),
        shared_already_computed_ROI=shared_roi,mean_event_retention_fraction=float(np.mean(retained)),
        original_ON_OFF_counts_exact=True,no_temporal_or_pixel_downsampling=True,
        timing_scope='Count preparation component only; alternating3 repetitions, not full FPS',
        timing_ms={n:dict(mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95))) for n,v in cost.items()})
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':run(sys.argv[1],sys.argv[2],shared_roi='--shared-roi' in sys.argv[3:])
