"""Combine the fixed acceleration mechanism with both measured prep schedules."""
from pathlib import Path
from collections import Counter
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from live_flow_acceleration import FlowAcceleration
from live_scale_rollout import scale_predictions
from summarize_live_pilot import reference_at,stats,read_rows
OUT=WORK/'output/flow_acceleration_20261026';assert not (OUT/'overlap_combination.json').exists()
PREV=WORK/'output/support_overlap_20261026';states={};pubs={};counts={}
for schedule,folder in [('serial','actual_serial'),('overlap','actual_overlap')]:
    model=FlowAcceleration();states[schedule]={};counts[schedule]=Counter()
    for row in read_rows(PREV/folder/'tracking.jsonl'):
        revised,diagnostic=model.step(row['t'],row['coefficient'],row['body'],row['diagnostic']['local_flow']['observation'],'position-consistent')
        states[schedule][round(row['t'],6)]=revised;counts[schedule][diagnostic['reason']]+=1
    pubs[schedule]={p['publication_slot']:p for p in read_rows(PREV/folder/'predictions.jsonl')}
records=sorted(json.loads((WORK/'output/dense_labels_side_110_1195/labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
tables={};samples=[];names=['v5_serial','new_acceleration_serial','new_acceleration_overlap']
for h in [.1,.2,.4]:
    errors={n:[] for n in names}
    for slot in sorted(set(pubs['serial'])&set(pubs['overlap'])):
        p={n:pubs[n][slot] for n in pubs};origin=max(r['t'] for r in p.values())
        if not all(r['valid'] and origin-r['observation_t']<=.16000001 and r['body_age_ms']/1000+origin-r['t']<=.16000001 for r in p.values()):continue
        truth=reference_at(records,origin+h)
        if truth is None:continue
        coefficients=dict(v5_serial=p['serial']['coefficients']['live_motion'],
            new_acceleration_serial=states['serial'][round(p['serial']['observation_t'],6)],
            new_acceleration_overlap=states['overlap'][round(p['overlap']['observation_t'],6)])
        points={};e={}
        for n,c in coefficients.items():
            schedule='overlap' if n.endswith('overlap') else 'serial';r=p[schedule]
            assert r['estimate_available_t']<=r['t']+1e-8<=origin+1e-8
            point=scale_predictions(c,origin-r['observation_t'])[0][str(h)]
            points[n]=point;e[n]=float(np.linalg.norm(np.asarray(point)-truth));errors[n].append(e[n])
        samples.append(dict(slot=slot,origin=origin,target_t=origin+h,horizon=h,truth=truth.tolist(),points=points,errors=e))
    tables[str(h)]={n:stats(e) for n,e in errors.items()}
report=dict(tables=tables,updates={n:dict(c) for n,c in counts.items()},samples=samples,
    virtual_common_origin=True,only_already_published_observation_prefixes=True,parameters_swept=False,RAW_or_GPU_started=False,
    scope='New fixed acceleration mechanism applied to both existing actual schedules; full combinations, not added independent gains.')
(OUT/'overlap_combination.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='samples'}),flush=True)
