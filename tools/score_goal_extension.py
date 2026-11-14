"""Keep the original and precommitted extension windows separate after inference."""
from pathlib import Path
import sys,json,numpy as np
from compare_final_pipeline import score


def main(run,label):
    path=Path(run);rows=[json.loads(s) for s in (path/'predictions.jsonl').read_text('utf-8').splitlines()]
    truth=sorted(json.loads(Path(label).read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    timing=[json.loads(s) for s in (path/'timings.jsonl').read_text('utf-8').splitlines()]
    summary=json.loads((path/'summary.json').read_text('utf-8'));accuracy=json.loads((path/'shared_motion_accuracy.json').read_text('utf-8'))
    scopes={}
    for name,lo,hi in [('fixed110_114',110.,114.),('extension114_118',114.,118.)]:
        r=[q for q in rows if lo<=q['t']<hi];times=[q for q in timing if lo<=q['t']<hi]
        if not r:continue
        sc=score(r,truth);last=[q['cost']['completion_lag_ms'] for q in r[-10:]]
        samples=times[1:] if lo==110. else times
        scopes[name]=dict(frames=len(r),fresh=sum(q['body'].get('detected',False) for q in r),valid=sum(q['valid'] for q in r),
            score={k:v for k,v in sc.items() if k!='anchors'},capacity_fps=1000/np.mean([q['total_service_ms'] for q in samples]),
            last10_lag_ms=dict(min=min(last),max=max(last),mean=float(np.mean(last))),
            max_completion_lag_ms=max(q['completion_lag_ms'] for q in times))
    result=dict(summary={k:summary[k] for k in ['frames','fresh_image_observations','valid_forecasts','steady_capacity_fps','paced_fresh_observation_rate','current_force_inputs','recent_amplitude_current_axis_inputs']},
        full_scores={k:v['physical_future_error_px'] for k,v in accuracy['scores'].items()},scopes=scopes,
        scope_frozen_before_inference=True,evaluation_scope_file='../side_extension_scope.json',
        no_change_to_original_scoring_window=True)
    (path/'extension_summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(*sys.argv[1:])
