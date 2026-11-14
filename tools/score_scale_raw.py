"""Score paired forecasts from actual RAW runs at their same real publication times."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
sys.path[:0]=[str(ROOT/'tools')]
from summarize_live_pilot import reference_at,stats,read_rows
OUT=WORK/'output/live_scale_rollout_RAW_20261026'
results={}
for case,label_case in [('side','side'),('side_v2','side'),('train2','train2')]:
    d=OUT/case;summary=json.loads((d/'completion.json').read_text('utf-8'))
    rr=read_rows(d/'tracking.jsonl');pp=read_rows(d/'predictions.jsonl')
    assert summary['sources_unchanged'] and summary['frame_descriptor_mismatches']==0
    assert summary['publication_slots']==len(pp) and summary['valid_publications']==sum(r['valid'] for r in pp)
    for row in rr:
        assert row['detector_frame_sha256']==row['gpu_frame_sha256']==row['producer_frame_sha256']
        assert row['max_event_t'] is None or row['max_event_t']<round(row['t']*1e6)
    for row in pp:
        assert row['estimate_available_t'] is None or row['estimate_available_t']<=row['t']+1e-8
        if row['valid']:assert row['observation_age_ms']<=160.00001 and row['body_age_ms']<=160.00001
    label=WORK/'output'/('dense_labels_side_110_1195' if label_case=='side' else 'dense_labels_train2_40_46')/'labels.json'
    records=sorted(json.loads(label.read_text('utf-8'))['records'].values(),key=lambda q:q['t'])
    tables={}
    for h in [.1,.2,.4]:
        errors={m:[] for m in ['unmodified_motion','live_motion']};changed=0;missing=0
        for row in pp:
            if not row['valid']:continue
            truth=reference_at(records,row['t']+h)
            if truth is None:missing+=1;continue
            for method in errors:
                point=row['predictions'][method][str(h)]
                errors[method].append(float(np.linalg.norm(np.asarray(point)-truth)))
            changed+=row['predictions']['unmodified_motion'][str(h)]!=row['predictions']['live_motion'][str(h)]
        tables[str(h)]=dict(metrics={m:stats(e) for m,e in errors.items()},changed=changed,missing_reference=missing)
    results[case]=dict(completion=summary,scores=tables,scale_qualified_tracking=sum(r['coefficient'] is not None and r['coefficient'].get('conditional_scale_rate') is not None for r in rr))
report=dict(scope='Actual RAW, paired baseline/scale at identical accepted origins and publication+h targets; known sparse manual interpolation<=150ms',
    results=results,default_promoted=False,blind_generalization_proven=False,
    first_side_stall_retained=True,stale_gate_speedup_not_established_by_sequential_runs=True,
    excluded_from_filtering=['manual labels','saved forecasts','saved detections'])
(OUT/'scores.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({n:dict(valid=r['completion']['valid_publications'],total=r['completion']['publication_slots'],scores=r['scores']) for n,r in results.items()}),flush=True)
