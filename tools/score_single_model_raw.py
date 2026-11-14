"""Actual single-model predictions and exact-frame offline original reference."""
from pathlib import Path
import json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_motion import LiveMotion,predictions
from summarize_live_pilot import reference_at,stats,read_rows
OUT=WORK/'output/single_model_optimized_RAW_20261026';results={}
window_summary=json.loads((WORK/'output/detector_window_v2_20261026/summary.json').read_text('utf-8'))
assert window_summary['model_sha256']=='e2531556ad0431d904fb0d1d432727a2e2dd9eaed9e098a6ed3d9e55f474db0b'
for case in ['side','train2']:
    folder=OUT/case;completion=json.loads((folder/'completion.json').read_text('utf-8'))
    tracking=read_rows(folder/'tracking.jsonl');pubs=read_rows(folder/'predictions.jsonl')
    assert completion['sources_unchanged'] and completion['native_batches']==205 and not completion['center_ensemble_enabled']
    assert len(pubs)==completion['publication_slots'] and sum(r['valid'] for r in pubs)==completion['valid_publications']
    references={};name='side_v2' if case=='side' else case
    for r in read_rows(WORK/'output/live_scale_rollout_RAW_20261026'/name/'tracking.jsonl'):
        references[round(r['t'],6)]=(r['body'],r['detector_frame_sha256'])
    # Existing control contains the first native-time tick even if the older
    # paced queue skipped it. All reference boxes are from the original model.
    control=json.loads((WORK/'output/detector_window_v2_20261026'/(case+'.json')).read_text('utf-8'))
    for r in control['rows']:
        boxes=r['windows']['20']['boxes']
        if boxes:
            best=max(boxes,key=lambda b:b['confidence']);b=np.asarray(best['bbox'])
            body=dict(t=r['t'],detected=True,bbox=b.tolist(),confidence=best['confidence'],center=((b[:2]+b[2:])/2).tolist())
        else:body=dict(t=r['t'],detected=False,center=None)
        key=round(r['t'],6);value=(body,r['windows']['20']['sha256'])
        if key in references:assert references[key]==value
        references[key]=value
    model=LiveMotion(25.);states={}
    for r in tracking:
        assert r['detector_frame_sha256']==r['producer_frame_sha256']==r['gpu_frame_sha256']
        assert r['max_event_t'] is None or r['max_event_t']<round(r['t']*1e6)
        original,digest=references[round(r['t'],6)];assert digest==r['detector_frame_sha256']
        c,valid,_=model.step(r['t'],original)
        states[round(r['t'],6)]=dict(coefficient=c,valid=valid,last_seen=model.seen)
    label=WORK/'output'/('dense_labels_side_110_1195' if case=='side' else 'dense_labels_train2_40_46')/'labels.json'
    records=sorted(json.loads(label.read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    scored=[];candidate_only=reference_only=0
    for r in pubs:
        assert r['estimate_available_t'] is None or r['estimate_available_t']<=r['t']+1e-8
        ref=states.get(round(r['observation_t'],6)) if r['observation_t'] is not None else None
        ref_valid=bool(ref and ref['valid'] and r['observation_age_ms']<=160.00001 and ref['last_seen'] is not None and r['t']-ref['last_seen']<=.16000001)
        candidate_only+=r['valid'] and not ref_valid;reference_only+=ref_valid and not r['valid']
        rr=dict(t=r['t'],valid=r['valid'],reference_valid=ref_valid,reference_forecasts=predictions(ref['coefficient'],r['observation_age_ms']/1000.)[0] if ref else {},predictions=r['predictions'])
        scored.append(rr)
    table={}
    for h in [.1,.2,.4]:
        errors={m:[] for m in ['original_model_replay','merged_curved','merged_scale']};missing=0
        for row in scored:
            if not(row['valid'] and row['reference_valid']):continue
            truth=reference_at(records,row['t']+h)
            if truth is None:missing+=1;continue
            points=dict(original_model_replay=row['reference_forecasts'][str(h)],
                merged_curved=row['predictions']['unmodified_motion'][str(h)],merged_scale=row['predictions']['live_motion'][str(h)])
            for method,p in points.items():errors[method].append(float(np.linalg.norm(np.asarray(p)-truth)))
        table[str(h)]=dict(metrics={m:stats(e) for m,e in errors.items()},missing_reference=missing)
    (folder/'scored_forecasts.json').write_text(json.dumps(scored,indent=2),encoding='utf-8')
    # Reuse the actual published preview; future markers are evaluation-only.
    cap=cv2.VideoCapture(str(folder/'predictions.mp4'))
    video=cv2.VideoWriter(str(folder/'comparison_04s.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5.,(1280,820));assert cap.isOpened() and video.isOpened()
    for i,(row,score) in enumerate(zip(pubs,scored)):
        ok,frame=cap.read();assert ok
        if row['valid']:
            origin=np.rint(row['coefficients']['live_motion']['position']).astype(int)
            for point,color in [(row['predictions']['live_motion']['0.4'],(255,180,20))]+([(score['reference_forecasts']['0.4'],(0,150,255))] if score['reference_valid'] else []):
                p=np.rint(point).astype(int);cv2.arrowedLine(frame,tuple(origin),tuple(p),color,2,tipLength=.12)
            truth=reference_at(records,row['t']+.4)
            if truth is not None:cv2.drawMarker(frame,tuple(np.rint(truth).astype(int)),(255,30,220),cv2.MARKER_CROSS,18,2)
        frame=cv2.copyMakeBorder(frame,0,100,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
        for j,text in enumerate(['ONE detector at runtime | blue=merged+scale | orange=offline original reference',
            f'pink=future reference (evaluation only); pub={row["t"]:.3f}s; valid={row["valid"]}']):
            cv2.putText(frame,text,(15,754+j*28),0,.57,(240,240,240),1)
        video.write(frame)
        if i in [5,10,15]:cv2.imwrite(str(folder/f'comparison_{i:02d}.png'),frame)
    assert not cap.read()[0];cap.release();video.release()
    results[case]=dict(completion=completion,scores=table,candidate_only_valid=int(candidate_only),reference_only_valid=int(reference_only),
        baseline_reference='Offline same-frame original NN/current state under candidate publication time; no wall-time baseline measurement.',
        exact_native_image_reference=True,extra_runtime_detector=False)
report=dict(results=results,default_promoted=False,reference_is_not_independently_timed=True,
    scope='Known4s clips, actual single-model runtime and same-target offline original comparison; no blind/generalized realtime claim.',
    source_cache_or_labels_used_during_actual_inference=False)
(OUT/'scores.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({case:dict(scores=r['scores'],valid=r['completion']['valid_publications'],total=r['completion']['publication_slots'],
    processing_ms=r['completion']['processing_median_ms'],age_p95_ms=r['completion']['observation_age_p95_ms'],
    candidate_only=r['candidate_only_valid'],reference_only=r['reference_only_valid']) for case,r in results.items()}),flush=True)
