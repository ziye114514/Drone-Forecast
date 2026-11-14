"""Fixed wall-source queries; include unavailable outputs in denominator."""
from pathlib import Path
import json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent/'output';sys.path.insert(0,str(ROOT))
from physical_publication import evaluate
from compare_final_pipeline import reference_at,stats

if __name__=='__main__':
    truth=sorted(json.loads((BASE/'separated201_20261107/reference13/references.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    output={}
    for path in ['native210_20261108/raw13_adaptive_native','arc215_20261108/raw13_adaptive_search',
        'uncertainty216_20261108/raw13_adaptive_uncertainty','live218_20261108/raw13_latest','process220_20261108/raw13_latest']:
        p=BASE/path
        if not (p/'summary.json').exists():continue
        rr=[json.loads(s) for s in (p/'predictions.jsonl').read_text('utf-8').splitlines()]
        pp=[json.loads(s) for s in (p/'publication_trace.jsonl').read_text('utf-8').splitlines()]
        records=[]
        for t in np.round(np.arange(10.8,15.6001,.2),1):
            recent=[r for r in pp if r['t']<=t+1e-9];r=recent[-1] if recent else None
            target=reference_at(truth,t+.4);available=bool(r and r['valid'] and t-r['t']<=.066667
                and t-r['source_t']<=.16 and r.get('last_image_t') is not None and t-r['last_image_t']<=.16 and target is not None)
            item=dict(t=float(t),valid=available,reason='scored' if available else 'no_recent_valid_published_forecast')
            if available:
                src=rr[r['index']];h=t-src['t']+.4;pred=evaluate(src,[h])[str(float(h))]
                item.update(error_px=float(np.linalg.norm(np.asarray(pred)-target)),prediction=pred,target=target.tolist(),
                    observation_age_s=t-src['t'],publication_age_s=t-r['t'])
                newer=[v for v in rr if v['t']<=t+1e-9]
                newest=newer[-1] if newer else None
                if newest and newest['valid'] and t-newest['t']<=.16:
                    h=t-newest['t']+.4;ideal=evaluate(newest,[h])[str(float(h))]
                    item['zero_compute_latency_same_dropped_packets_error_px']=float(np.linalg.norm(np.asarray(ideal)-target))
            records.append(item)
        output[path]=dict(eligible=25,available=sum(v['valid'] for v in records),
            error=stats([v['error_px'] for v in records if v['valid']]),records=records,
            actual_publication_time_used=True,missing_outputs_kept=True,reference_is_weak_event_outline=True,
            publication_reuse_max_age_s=.066667,state_and_last_image_max_age_s=.16)
        paired=[v for v in records if v['valid'] and 'zero_compute_latency_same_dropped_packets_error_px' in v]
        output[path]['latency_counterfactual_same_queries']=dict(n=len(paired),
            actual=stats([v['error_px'] for v in paired]),
            hypothetical_zero_compute_latency=stats([v['zero_compute_latency_same_dropped_packets_error_px'] for v in paired]),
            not_a_real_time_result=True,no_future_source_events_used=True)
    dest=BASE/'live218_20261108';dest.mkdir(exist_ok=True)
    (dest/'publication_scores.json').write_text(json.dumps(output,indent=2),'utf-8')
    print({k:{'available':v['available'],'error':v['error']} for k,v in output.items()})
