"""Score frozen search/transport outputs without changing source origins.

Labels are opened only here, after inference. Misses remain in records.
The ready-time horizon audit uses the paced source clock, not the GPU FPS.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'tools')]
from compare_final_pipeline import reference_at,score,stats
from diagnose_maneuver_horizon import SPECS
from score_fred_adaptation import matched


def load(path):
    return [json.loads(x) for x in Path(path).read_text('utf-8').splitlines()]


def audit(run,name):
    _,original,label=next(s for s in SPECS if s[0]==name)
    out=ROOT.parent/'output'
    rows=load(run/'predictions.jsonl');old=load(out/original)
    previous=load(out/'maneuver_response_20261029'/
        ('RAW_side_axis_rebase8s' if name=='side' else 'RAW_other_axis_rebase6s')/'predictions.jsonl')
    labels=sorted(json.loads((out/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    pair=matched(old,rows,lambda t:reference_at(labels,t))
    by_time={round(r['t'],6):r for r in previous};retained=[]
    for p in pair['records']:
        r=by_time.get(round(p['t'],6))
        if not p['available'] or r is None or not r['valid']:continue
        target=reference_at(labels,p['t']+.4)
        retained.append(dict(t=p['t'],previous_error_px=float(np.linalg.norm(np.asarray(r['physical']['0.4'])-target)),
            new_error_px=p['new_error_px']))
    costs=[r['cost']['completion_lag_ms'] for r in rows if r['valid']]
    recent=None;delayed=[];discards=[]
    for r in rows:
        a=r.get('nn_acquisition_schedule') or {}
        if a.get('deferred') and recent is not None and r['t']-recent<=.3+1e-8:delayed.append(r['t'])
        if r['body'].get('detected'):recent=r['t']
        if r.get('stale_NN_reply_discarded_for_current_recovery'):discards.append(r['t'])
    return dict(run=str(run.resolve()),runtime=json.loads((run/'summary.json').read_text()),
        matched_original=pair,manual_score=score(rows,labels),
        previous_actual_on_common_retained_origins=dict(n=len(retained),
            previous=stats([p['previous_error_px'] for p in retained]),new=stats([p['new_error_px'] for p in retained]),records=retained),
        valid_source_forecast_expired_fraction=None if not costs else float(np.mean(np.asarray(costs)>=400)),
        remaining_400ms_horizon_at_ready_ms=stats([400-x for x in costs]),
        remaining_horizon_p05_ms=None if not costs else float(np.percentile(400-np.asarray(costs),5)),
        recent_body_loss_deferred=delayed,pending_replies_discarded=discards,
        inference_uses_labels=False,score_scope='Exposed development labels, fixed original source times; no full-dataset/generalization claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--name',choices=['side','other'],required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=audit(args.run,args.name)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    print(json.dumps(dict(matched={k:v for k,v in result['matched_original'].items() if k!='records'},
        paired={k:v for k,v in result['previous_actual_on_common_retained_origins'].items() if k!='records'},
        expired=result['valid_source_forecast_expired_fraction'],recent_loss_deferred=result['recent_body_loss_deferred'],
        discards=len(result['pending_replies_discarded'])),indent=2))
