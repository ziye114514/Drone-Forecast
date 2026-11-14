"""One fixed equal ensemble of already computed, same-frame model observations."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from live_center_ensemble import blend_current,DESIGN
from live_motion import LiveMotion,predictions
from summarize_live_pilot import reference_at,stats,read_rows
OUT=WORK/'output/equal_center_ensemble_20261026';OUT.mkdir(exist_ok=False)
(OUT/'live_center_ensemble.py').write_bytes((ROOT/'live_center_ensemble.py').read_bytes())
(OUT/'design.json').write_text(json.dumps(dict(**DESIGN,
    motivation='Whole adapted model mixed results; try one standard equal ensemble, original starts/observability preserved. Exploratory new mechanism, no weight sweep.',
    gate='Both .4s means decrease; .1s neither worsens more than1px; all original valid targets retained.'),indent=2),encoding='utf-8')
assert blend_current(dict(detected=False),[dict(bbox=[0,0,10,10],confidence=1.)])[0]['detected'] is False
results={}
for case,label_case in [('side_v2','side'),('train2','train2')]:
    probe=json.loads((WORK/'output/roi_augmented_detector_20261026/forecast_probe'/(case+'.json')).read_text('utf-8'))
    proposals={round(r['t'],6):r['adapted_body'] for r in probe['rows']}
    run=WORK/'output/live_scale_rollout_RAW_20261026'/case;inputs=read_rows(run/'tracking.jsonl');pubs=read_rows(run/'predictions.jsonl')
    models={name:LiveMotion(25.) for name in ['baseline','ensemble']};states={name:{} for name in models};traces=[];used=0
    for row in inputs:
        p=proposals[round(row['t'],6)];candidate,audit=blend_current(row['body'],[p] if p['detected'] else [])
        used+=audit['used'];current={}
        for name,model in models.items():
            body=row['body'] if name=='baseline' else candidate
            c,valid,d=model.step(row['t'],body);states[name][round(row['t'],6)]=c;current[name]=(c,valid)
            if name=='baseline' and c is not None:
                for field in ['position','velocity','acceleration']:assert c[field]==row['coefficient'][field]
        assert current['baseline'][1]==current['ensemble'][1]
        assert candidate['detected']==row['body']['detected']
        if candidate['detected']:assert candidate['bbox']==row['body']['bbox']
        traces.append(dict(t=row['t'],body=candidate,audit=audit,coefficient=current['ensemble'][0]))
    records=sorted(json.loads((WORK/'output'/('dense_labels_side_110_1195' if label_case=='side' else 'dense_labels_train2_40_46')/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    table={}
    for h in [.1,.2,.4]:
        errors={name:[] for name in models}
        for pub in pubs:
            if not pub['valid']:continue
            truth=reference_at(records,pub['t']+h)
            if truth is None:continue
            for name in models:
                c=states[name][round(pub['observation_t'],6)]
                point=predictions(c,pub['observation_age_ms']/1000.)[0][str(h)]
                errors[name].append(float(np.linalg.norm(np.asarray(point)-truth)))
        table[str(h)]={name:stats(e) for name,e in errors.items()}
    results[case]=dict(used=used,tracking=len(inputs),scores=table,original_detections_boxes_and_motion_validity_exact=True)
    (OUT/(case+'.json')).write_text(json.dumps(traces,indent=2),encoding='utf-8')
passed=all(r['scores']['0.4']['ensemble']['mean']<r['scores']['0.4']['baseline']['mean'] and
    r['scores']['0.1']['ensemble']['mean']<=r['scores']['0.1']['baseline']['mean']+1 for r in results.values())
report=dict(results=results,provisional_gate_passed=passed,additional_inference_latency_accounted_for=False,
    design_after_previous_direct_model_results=True,globally_blind=False,default_promoted=False)
(OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
