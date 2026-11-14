"""Align already-published snapshots at the later source-time origin per slot."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_scale_rollout import scale_predictions
from summarize_live_pilot import read_rows,reference_at,stats
OUT=WORK/'output/support_overlap_20261026';assert not (OUT/'common_origin_comparison.json').exists()
runs={n:{p['publication_slot']:p for p in read_rows(OUT/folder/'predictions.jsonl')} for n,folder in [('serial','actual_serial'),('overlap','actual_overlap')]}
records=sorted(json.loads((WORK/'output/dense_labels_side_110_1195/labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
tables={};samples=[];excluded=[]
for h in [.1,.2,.4]:
    errors={n:[] for n in runs}
    for slot in sorted(set.intersection(*(set(r) for r in runs.values()))):
        rows={n:r[slot] for n,r in runs.items()};origin=max(p['t'] for p in rows.values())
        valid=all(p['valid'] and p['observation_t'] is not None and origin-p['observation_t']<=.16000001
            and p['body_age_ms']/1000+origin-p['t']<=.16000001 for p in rows.values())
        if not valid:
            if h==.4:excluded.append(slot)
            continue
        truth=reference_at(records,origin+h)
        if truth is None:continue
        points={};e={}
        for n,p in rows.items():
            assert p['estimate_available_t']<=p['t']+1e-8<=origin+1e-8
            c=p['coefficients']['live_motion'];point=scale_predictions(c,origin-p['observation_t'])[0][str(h)]
            e[n]=float(np.linalg.norm(np.asarray(point)-truth));errors[n].append(e[n]);points[n]=point
        samples.append(dict(slot=slot,common_origin=origin,target_t=origin+h,horizon=h,truth=truth.tolist(),points=points,errors=e,
            original_publications={n:p['t'] for n,p in rows.items()},extra_wait_ms={n:(origin-p['t'])*1000 for n,p in rows.items()}))
    tables[str(h)]={n:stats(e) for n,e in errors.items()}
policy=json.loads((ROOT/'configs/accumulation_policy.json').read_text('utf-8'));failures=[]
for h,s in tables.items():
    for field,key in [('mean','mean_regression_allowance'),('p95','p95_regression_allowance')]:
        rule=policy['accuracy'][key];a=s['serial'][field];b=s['overlap'][field]
        allowance=max(rule['absolute_px'],rule['relative_fraction']*a)
        if b>a+allowance:failures.append(f'{h}:{field}:{b-a:.6f}>{allowance:.6f}')
result=dict(tables=tables,guard_failures=failures,excluded_invalid_slots_04s=excluded,samples=samples,
    snapshots_published_before_common_origin=True,no_newer_or_future_observation_selected=True,
    virtual_common_origin=True,scope='Later of two actual publications in each same source-time slot, using only already-published snapshots. Fair common target comparison, not both original live deadlines or a globally blind benchmark.')
(OUT/'common_origin_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)
