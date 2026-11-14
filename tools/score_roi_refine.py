"""Score frozen-time ROI center candidate; labels are opened only here."""
from pathlib import Path
import copy,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_motion import LiveMotion,predictions
from summarize_live_pilot import reference_at,stats,read_rows
OUT=WORK/'output/roi_refine_20261026';results={}
for case,label_case in [('side_v2','side'),('train2','train2')]:
    probe=json.loads((OUT/(case+'.json')).read_text('utf-8'));by_t={round(r['t'],6):r for r in probe['rows']}
    run=WORK/'output/live_scale_rollout_RAW_20261026'/case
    inputs=read_rows(run/'tracking.jsonl');pubs=read_rows(run/'predictions.jsonl')
    models=dict(baseline=LiveMotion(25.),refined=LiveMotion(25.));states={k:{} for k in models};traces=[]
    for row in inputs:
        r=by_t[round(row['t'],6)];current={}
        for name,model in models.items():
            body=copy.deepcopy(row['body'])
            if name=='refined' and r['refined_body'] is not None:
                box=np.asarray(r['refined_body']['bbox']);body['center']=((box[:2]+box[2:])/2).tolist()
            c,valid,d=model.step(row['t'],body);current[name]=(c,valid)
            states[name][round(row['t'],6)]=c
            if name=='baseline' and c is not None:
                for field in ['position','velocity','acceleration']:assert c[field]==row['coefficient'][field]
        assert current['baseline'][1]==current['refined'][1]
        traces.append(dict(t=row['t'],baseline=current['baseline'][0],refined=current['refined'][0]))
    records=sorted(json.loads((WORK/'output'/('dense_labels_side_110_1195' if label_case=='side' else 'dense_labels_train2_40_46')/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    tables={}
    for h in [.1,.2,.4]:
        errors={name:[] for name in models};changed=0
        for pub in pubs:
            if not pub['valid']:continue
            truth=reference_at(records,pub['t']+h)
            if truth is None:continue
            points={}
            for name in models:
                c=states[name][round(pub['observation_t'],6)]
                point=predictions(c,pub['observation_age_ms']/1000.)[0][str(h)]
                points[name]=point;errors[name].append(float(np.linalg.norm(np.asarray(point)-truth)))
            changed+=points['baseline']!=points['refined']
        tables[str(h)]=dict(metrics={name:stats(e) for name,e in errors.items()},changed_predictions=changed)
    center={name:[] for name in models}
    for trace in traces:
        truth=reference_at(records,trace['t'])
        if truth is None:continue
        for name in models:
            c=trace[name]
            if c is not None:center[name].append(float(np.linalg.norm(np.asarray(c['position'])-truth)))
    results[case]=dict(current_center={name:stats(e) for name,e in center.items()},scores=tables,
        refined_ticks=probe['refined_centers'],observed_NN_validity_unchanged=True)
    (OUT/(case+'_states.json')).write_text(json.dumps(traces,indent=2),encoding='utf-8')
passed=all(results[name]['scores']['0.4']['metrics']['refined']['mean']<results[name]['scores']['0.4']['metrics']['baseline']['mean'] and
    results[name]['scores']['0.1']['metrics']['refined']['mean']<=results[name]['scores']['0.1']['metrics']['baseline']['mean']+1 for name in results)
report=dict(results=results,provisional_gate_passed=passed,default_promoted=False,
    second_inference_latency_accounted_for=False,
    scope='Same accepted NN cutoffs and frozen publication targets; center candidate only. Not paced deployment or blind generalization.')
(OUT/'scores.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
