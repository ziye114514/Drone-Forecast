"""Independent scoring of both prespecified adapted-detector variants."""
from pathlib import Path
import argparse,copy,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_motion import LiveMotion,predictions
from summarize_live_pilot import reference_at,stats,read_rows
parser=argparse.ArgumentParser();parser.add_argument('--model-dir',type=Path);args=parser.parse_args()
OUT=(args.model_dir.resolve() if args.model_dir else WORK/'output/roi_augmented_detector_20261026')/'forecast_probe';results={}
for case,label_case in [('side_v2','side'),('train2','train2')]:
    probe=json.loads((OUT/(case+'.json')).read_text('utf-8'));by_t={round(r['t'],6):r for r in probe['rows']}
    run=WORK/'output/live_scale_rollout_RAW_20261026'/case;inputs=read_rows(run/'tracking.jsonl');pubs=read_rows(run/'predictions.jsonl')
    names=['baseline','adapted_full']+(['adapted_refined'] if probe.get('ROI_requested',True) else [])
    models={name:LiveMotion(25.) for name in names};states={k:{} for k in models}
    for row in inputs:
        r=by_t[round(row['t'],6)]
        for name,model in models.items():
            body=copy.deepcopy(row['body'] if name=='baseline' else r['adapted_body'])
            if name=='adapted_refined' and r['refined_body'] is not None:
                b=np.asarray(r['refined_body']['bbox']);body['center']=((b[:2]+b[2:])/2).tolist()
            c,valid,d=model.step(row['t'],body);states[name][round(row['t'],6)]=dict(coefficient=c,valid=valid,last_seen=model.seen)
            if name=='baseline' and c is not None:
                for field in ['position','velocity','acceleration']:assert c[field]==row['coefficient'][field]
    records=sorted(json.loads((WORK/'output'/('dense_labels_side_110_1195' if label_case=='side' else 'dense_labels_train2_40_46')/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    table={}
    for h in [.1,.2,.4]:
        scores={name:[] for name in models};original_targets=0;discarded=0
        for pub in pubs:
            if not pub['valid']:continue
            truth=reference_at(records,pub['t']+h)
            if truth is None:continue
            original_targets+=1;current={name:states[name][round(pub['observation_t'],6)] for name in models}
            good=all(v['valid'] and v['last_seen'] is not None and pub['t']-v['last_seen']<=.16+1e-8 for v in current.values())
            if not good:discarded+=1;continue
            for name,v in current.items():
                point=predictions(v['coefficient'],pub['observation_age_ms']/1000.)[0][str(h)]
                scores[name].append(float(np.linalg.norm(np.asarray(point)-truth)))
        table[str(h)]=dict(original_targets=original_targets,discarded_for_common_validity=discarded,
            metrics={name:stats(e) for name,e in scores.items()})
    results[case]=dict(scores=table,inputs=probe['matched_current_ticks'],detections=probe['full_detections'],
        original_detections=probe['original_detections'],refined_centers=probe['refined_centers'])
gates={}
for name in [name for name in models if name!='baseline']:
    gates[name]=all(r['scores']['0.4']['discarded_for_common_validity']==0 and
        r['scores']['0.4']['metrics'][name]['mean']<r['scores']['0.4']['metrics']['baseline']['mean'] and
        r['scores']['0.1']['metrics'][name]['mean']<=r['scores']['0.1']['metrics']['baseline']['mean']+1 for r in results.values())
report=dict(results=results,provisional_gates=gates,default_promoted=False,additional_latency_accounted_for=False,
    globally_blind_data=False,no_hyperparameter_sweep=True)
(OUT/'scores.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
