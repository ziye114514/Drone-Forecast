"""Compare retimed and stale predictions at identical actual output times.

Sixth deadline of each30Hz group is fixed before opening exposed labels. This
scores actual publication+0.4, not the older source-observation+0.4 endpoint.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'tools')]
from compare_final_pipeline import reference_at,stats


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    rows=[json.loads(x) for x in (args.run/'predictions.jsonl').read_text().splitlines()]
    by_time={r['t']:r for r in rows}
    pubs=[json.loads(x) for x in (args.run/'publications.jsonl').read_text().splitlines()]
    labels=sorted(json.loads(args.labels.read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    records=[]
    for pub in pubs:
        if pub['index']%6:continue
        truth=reference_at(labels,pub['t']+.4)
        if truth is None:continue
        item=dict(index=pub['index'],t=pub['t'],source_t=pub['source_t'],valid=pub['valid'],age_s=pub['source_age_s'])
        if pub['valid']:
            r=by_time[pub['source_t']]
            item.update(stale_error_px=float(np.linalg.norm(np.asarray(r['physical']['0.4'])-truth)),
                retimed_error_px=float(np.linalg.norm(np.asarray(pub['physical']['0.4'])-truth)))
        records.append(item)
    good=[r for r in records if r['valid']]
    result=dict(target_slots=len(records),available=len(good),stale=stats([r['stale_error_px'] for r in good]),
        retimed=stats([r['retimed_error_px'] for r in good]),records=records,
        same_current_state_and_actual_target_times=True,inference_uses_labels=False,
        endpoint='actual publication time+0.4s',view_verified=False)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
