from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from compare_final_pipeline import stats
OUT=ROOT.parent/'output/decision169_20261105'
if __name__=='__main__':
    completion=json.loads((OUT/'fit_completion.json').read_text('utf-8'));assert not completion['source_changed_during_run']
    predictions=json.loads((OUT/'heldout_predictions.json').read_text('utf-8'));test=json.loads((OUT/'heldout_targets.json').read_text('utf-8'));summary={}
    for name,arms in predictions.items():
        answer={};pairs={};targets={r['query_t']:r for r in test[name]}
        for arm,rows in arms.items():
            records=[]
            for r in rows:
                gt=targets[r['query_t']];valid=r['valid'] and gt['target'] is not None
                records.append(dict(r,target=gt['target'],scored=valid,
                    error_px=float(np.linalg.norm(np.asarray(r['prediction'])-gt['target'])) if valid else None,
                    baseline_error_px=float(np.linalg.norm(np.asarray(r['baseline'])-gt['target'])) if valid else None))
            answer[arm]=dict(available=sum(r['scored'] for r in records),planned=len(records),
                error=stats([r['error_px'] for r in records if r['scored']]),records=records)
        base=[r['baseline_error_px'] for r in answer['zero']['records'] if r['scored']]
        for control in ['zero','past_delayed']:
            p=[(a,b) for a,b in zip(answer['aligned']['records'],answer[control]['records']) if a['scored'] and b['scored']]
            delta=[b['error_px']-a['error_px'] for a,b in p]
            pairs[control]=dict(common=len(p),mean_reduction_px=float(np.mean(delta)) if delta else None,
                improved=sum(d>1e-6 for d in delta),worsened=sum(d< -1e-6 for d in delta))
        summary[name]=dict(arms=answer,uncalibrated_baseline=stats(base),comparisons=pairs)
        print(name,json.dumps(dict(baseline=stats(base),**{a:v['error'] for a,v in answer.items()})),flush=True)
    summary['decision']=dict(all_suffixes_beat_both_controls=all(all(v['mean_reduction_px'] is not None and v['mean_reduction_px']>0 for v in c['comparisons'].values()) for k,c in summary.items()),
        independent_new_test=False,manual_calibration_diagnostic=True,universal_upper_bound=False,more_parameter_search_allowed_in_this_batch=False)
    (OUT/'decision_scores.json').write_text(json.dumps(summary,indent=2),'utf-8')
