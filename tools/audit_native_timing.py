"""Compare native temporal resolution and per-pixel support, no annotation input."""
from pathlib import Path
import sys,json,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_decode import native_batches


def main():
    base=ROOT.parent/'output';out=base/'stability_audit_20261031/native_timing';out.mkdir(exist_ok=False)
    cases=[('clear',26.,'E:/dataset/train/6/media/event.raw','target25_continuity_20261031/RAW_clear_center4s'),
        ('side',110.,'E:/dataset/train/6/media/event.raw','reacquisition_motion_20261029/RAW_side_default8s'),
        ('train2',43.,'E:/dataset/train/2/aligned/event.raw','target25_continuity_20261031/RAW_other_center6s')]
    result={}
    for name,start,raw,folder in cases:
        rows=[json.loads(s) for s in (base/folder/'predictions.jsonl').read_text('utf-8').splitlines()]
        begin=(int(start*1e6)//33333)*33333/1e6;stamps=[];roi=[];count=0
        for t,e in native_batches(Path(raw),begin,1.,False,33333):
            stamps.append(e['t'].copy());count+=len(e)
            row=min(rows,key=lambda r:abs(r['t']-t));b=row['body'].get('bbox')
            if b is None:continue
            keep=(e['x']>=b[0])&(e['x']<b[2])&(e['y']>=b[1])&(e['y']<b[3]);roi.append(e[keep].copy())
        ts=np.concatenate(stamps);u=np.unique(ts);diff=np.diff(u)
        value=dict(native_events=count,unique_timestamps=len(u),distinct_positive_timestamp_gaps_us={str(k):int(v) for k,v in collections.Counter(diff).most_common(8)},
            timestamp_remainder_diversity={str(n):len(np.unique(u%n)) for n in (10,100,1000)},
            no_prediction_labels_read=True)
        if roi:
            e=np.concatenate(roi);cutoff=int(begin*1e6)+1000000;e=e[(e['t']>=cutoff-160000)&(e['t']<cutoff)]
            pixel=e['y'].astype(np.int64)*1280+e['x'];ids,counts=np.unique(pixel,return_counts=True)
            value.update(last160ms_body_events=len(e),active_pixels=len(ids),pixels_with_at_least14_events=int(sum(counts>=14)),
                events_per_pixel_median=float(np.median(counts)) if len(counts) else None,
                events_per_pixel_p95=float(np.percentile(counts,95)) if len(counts) else None)
            np.savez_compressed(out/(name+'_160ms.npz'),events=e,cutoff=cutoff/1e6)
        result[name]=value
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
