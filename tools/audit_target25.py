"""Frozen-output accuracy and causal, fresh30Hz continuity audit.

No held/repeated forecasts count as current-image observations. Algorithmic
axis_only is not a human-verified view label; unknown is never relabelled front.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'tools')]
from compare_final_pipeline import score,reference_at,stats
from score_fred_adaptation import matched


def read(run):return [json.loads(s) for s in (run/'predictions.jsonl').read_text('utf-8').splitlines()]


def continuity(rows,hz=30.):
    # Run lengths require a genuine current image and a fresh valid state,
    # completed in its OWN source period. Late catch-up is not timely30Hz.
    segments=[];start=previous=None;n=0
    for r in rows:
        good=bool(r['body'].get('detected') and r['valid'] and r['cost']['completion_lag_ms']<=1000/hz)
        adjacent=previous is not None and abs(r['t']-previous-1/hz)<2e-5
        if not good or (n and not adjacent):
            if n:segments.append(dict(first_t=start,last_t=previous,states=n,source_span_s=previous-start))
            n=0
        if good:
            if not n:start=r['t']
            n+=1
        previous=r['t']
    if n:segments.append(dict(first_t=start,last_t=previous,states=n,source_span_s=previous-start))
    gaps=[]
    for a,b in zip(rows,rows[1:]):
        if a['valid'] and b['valid'] and a['body'].get('detected') and b['body'].get('detected'):
            gaps.append(1000*(b['t']-a['t'])+b['cost']['completion_lag_ms']-a['cost']['completion_lag_ms'])
    return dict(longest_timely_fresh30Hz_segment=max(segments,key=lambda x:x['states']) if segments else None,
        all_timely_segments=segments,ready_interarrival_ms=stats(gaps),
        valid_current_image_states=sum(r['valid'] and r['body'].get('detected',False) for r in rows),
        missed_or_invalid_source_ticks=sum(not r['valid'] or not r['body'].get('detected',False) for r in rows),
        fixed_source_rate_hz=hz,repeated_outputs_counted_as_new_observations=False)


def main(args):
    a,b=read(args.reference),read(args.candidate)
    if [r['t'] for r in a]!=[r['t'] for r in b]:raise ValueError('Different source cutoffs')
    result=dict(source_ticks=len(a),runtime_labels_used=False,
        reference=dict(summary=json.loads((args.reference/'summary.json').read_text()),continuity=continuity(a)),
        candidate=dict(summary=json.loads((args.candidate/'summary.json').read_text()),continuity=continuity(b)),
        rigid_center_applied=sum((r['tracking'].get('rigid_center_transport') or {}).get('applied',False) for r in b),
        views='algorithmic modes only; not verified frontal/side ground truth')
    if args.labels:
        labels=sorted(json.loads(args.labels.read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        result.update(matched=matched(a,b,lambda t:reference_at(labels,t)),
            reference_score=score(a,labels),candidate_score=score(b,labels))
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps(dict(matched={k:v for k,v in result.get('matched',{}).items() if k!='records'},
        applied=result['rigid_center_applied'],timely={k:v['continuity']['longest_timely_fresh30Hz_segment'] for k,v in result.items() if k in ('reference','candidate')}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--labels',type=Path);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
