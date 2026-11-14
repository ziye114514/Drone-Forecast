"""Existing sparse test3 boxes as exact targets at reported variable horizons."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from live_motion import LiveMotion,curved_forecast
from live_scale_rollout import scale_predictions
from summarize_live_pilot import stats,read_rows
OUT=WORK/'output/heldout_motion_RAW_20261026'
current=read_rows(OUT/'tracking.jsonl');pubs=read_rows(OUT/'predictions.jsonl')
reference=json.loads((OUT/'original_reference/observations.json').read_text('utf-8'));by_t={round(r['t'],6):r for r in reference}
model=LiveMotion(25.);states={}
for row in current:
    r=by_t[round(row['t'],6)];assert r['frame_sha256']==row['detector_frame_sha256']
    c,valid,_=model.step(row['t'],r['body']);states[round(row['t'],6)]=dict(coefficient=c,valid=valid,last_seen=model.seen)
labels=json.loads((WORK/'output/raw_only_project/evaluation_data/train3.json').read_text('utf-8'))['labels']
errors={name:[] for name in ['original_reference','merged_curved','merged_scale']};samples=[];unavailable=unmatched=0
for label in labels:
    t=float(label['t'])
    if label['state']!='present' or not .3<=t<=8:continue
    possible=[p for p in pubs if .3-1e-8<=t-p['t']<=.5+1e-8]
    if not possible:unmatched+=1;continue
    # Select by time before checking any model's error or validity.
    pub=min(possible,key=lambda p:(abs(t-p['t']-.4),p['t']))
    ref=states.get(round(pub['observation_t'],6)) if pub['observation_t'] is not None else None
    good=bool(ref and ref['valid'] and ref['last_seen'] is not None and pub['t']-ref['last_seen']<=.16000001)
    if not(pub['valid'] and good):unavailable+=1;continue
    H=t-pub['observation_t'];c=pub['coefficients']['live_motion'];truth_box=np.asarray(label['bbox']);truth=(truth_box[:2]+truth_box[2:])/2
    points=dict(original_reference=curved_forecast(ref['coefficient'],H),merged_curved=curved_forecast(c,H),
                merged_scale=scale_predictions(c,H-.1)[0]['0.1'])
    sample=dict(target_t=t,publication_t=pub['t'],horizon_s=t-pub['t'],truth=truth.tolist(),predictions=points)
    for name,p in points.items():errors[name].append(float(np.linalg.norm(np.asarray(p)-truth)))
    samples.append(sample)
completion=json.loads((OUT/'completion.json').read_text('utf-8'))
assert completion['sources_unchanged'] and completion['native_batches']==400
report=dict(scope='One preselected0–8s original test3 clip. Existing sparse weak boxes; each exact target uses nearest .4s publication within .3–.5s, then common validity. NOT fixed0.4s error.',
    metrics={name:stats(e) for name,e in errors.items()},samples=samples,
    actual_horizon_range_s=[min(r['horizon_s'] for r in samples),max(r['horizon_s'] for r in samples)] if samples else None,
    targets_unavailable=unavailable,targets_without_time_origin=unmatched,
    completion=completion,baseline_is_offline_same_frame_reference=True,default_promoted=False,
    training_test_separated_but_historically_exposed=True,labels_not_used_in_runtime=True)
(OUT/'scores.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['samples','completion']}),flush=True)
