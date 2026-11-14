"""Paired actual publication targets for original/ensemble states, plus video."""
from pathlib import Path
import json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_motion import LiveMotion
from summarize_live_pilot import reference_at,stats,read_rows
OUT=WORK/'output/center_ensemble_RAW_20261026';results={}
for case in ['side','train2']:
    folder=OUT/case;completion=json.loads((folder/'completion.json').read_text('utf-8'))
    tracking=read_rows(folder/'tracking.jsonl');pubs=read_rows(folder/'predictions.jsonl')
    assert completion['sources_unchanged'] and completion['native_batches']==205
    assert len(tracking)==completion['tracking_updates'] and len(pubs)==completion['publication_slots']
    assert sum(r['valid'] for r in pubs)==completion['valid_publications']
    baseline=LiveMotion(25.)
    for row in tracking:
        assert row['producer_frame_sha256']==row['detector_frame_sha256']==row['gpu_frame_sha256']
        assert row['max_event_t'] is None or row['max_event_t']<round(row['t']*1e6)
        assert row['body']['detected']==row['original_body']['detected']
        if row['body']['detected']:assert row['body']['bbox']==row['original_body']['bbox']
        coefficient,valid,_=baseline.step(row['t'],row['original_body'])
        assert coefficient==row['full_frame_baseline'] and valid==row['valid']
    label=WORK/'output'/('dense_labels_side_110_1195' if case=='side' else 'dense_labels_train2_40_46')/'labels.json'
    records=sorted(json.loads(label.read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    table={}
    for h in [.1,.2,.4]:
        errors={m:[] for m in ['full_frame_baseline','live_motion']};missing=0
        for row in pubs:
            assert row['valid']==row['full_frame_baseline_valid']
            assert row['estimate_available_t'] is None or row['estimate_available_t']<=row['t']+1e-8
            if not row['valid']:continue
            assert row['observation_age_ms']<=160.00001 and row['body_age_ms']<=160.00001
            truth=reference_at(records,row['t']+h)
            if truth is None:missing+=1;continue
            for method in errors:
                point=row['predictions'][method][str(h)]
                errors[method].append(float(np.linalg.norm(np.asarray(point)-truth)))
        table[str(h)]=dict(metrics={m:stats(e) for m,e in errors.items()},missing_reference=missing)
    results[case]=dict(scores=table,completion=completion,
        ensemble_current_ticks=sum(bool((r.get('center_ensemble') or {}).get('used')) for r in tracking),
        original_observability_and_same_frame_baseline_exact=True)
    # Postprocessing only: future reference is never read by inference.
    cap=cv2.VideoCapture(str(folder/'predictions.mp4'))
    video=cv2.VideoWriter(str(folder/'paired_04s.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5.,(1280,820))
    assert cap.isOpened() and video.isOpened();sampled=[]
    for i,row in enumerate(pubs):
        ok,frame=cap.read();assert ok
        if row['valid']:
            origin=np.rint(row['coefficients']['live_motion']['position']).astype(int)
            for method,color,name in [('full_frame_baseline',(0,150,255),'original'),('live_motion',(255,160,20),'ensemble')]:
                point=np.rint(row['predictions'][method]['0.4']).astype(int)
                cv2.arrowedLine(frame,tuple(origin),tuple(point),color,2,tipLength=.12)
            truth=reference_at(records,row['t']+.4)
            if truth is not None:
                p=np.rint(truth).astype(int);cv2.drawMarker(frame,tuple(p),(255,30,220),cv2.MARKER_CROSS,18,2)
        frame=cv2.copyMakeBorder(frame,0,100,0,0,cv2.BORDER_CONSTANT,value=(22,22,22))
        for j,text in enumerate(['0.4s: orange=original | blue=equal ensemble | pink=future reference (evaluation only)',
            f'publication={row["t"]:.3f}s; valid={row["valid"]}; additional model latency INCLUDED']):
            cv2.putText(frame,text,(15,757+j*27),0,.58,(230,230,230),1)
        video.write(frame)
        if i in [len(pubs)//4,len(pubs)//2,3*len(pubs)//4]:
            image=folder/f'paired_{i:02d}.png';cv2.imwrite(str(image),frame);sampled.append(str(image))
    assert not cap.read()[0];cap.release();video.release()
    results[case]['postprocessed_video_frames']=len(pubs)
    results[case]['sampled_frames']=sampled
report=dict(scope='Actual paced RAW, two models included in latency, same accepted input history and actual publication+h targets.',
    results=results,manual_reference='Sparse present-box-center interpolation only across<=150ms, previously exposed clips.',
    primary_baseline_always_retained=True,default_promoted=False,mechanical_RPM_or_thrust_used=False,
    SOTA_or_blind_generalization_proven=False)
(OUT/'scores.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({case:dict(scores=r['scores'],valid=r['completion']['valid_publications'],slots=r['completion']['publication_slots'],
    age_p95_ms=r['completion']['observation_age_p95_ms'],processing_median_ms=r['completion']['processing_median_ms'],
    tracking=r['completion']['tracking_updates'],ensemble_ticks=r['ensemble_current_ticks']) for case,r in results.items()}),flush=True)
