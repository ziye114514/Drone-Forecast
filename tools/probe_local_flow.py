"""Native event sparse flow and ON/OFF-support LK at frozen accepted times."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT)]
from live_decode import native_batches
from live_local_flow import FeatureFlow,event_support,summarize_sparse,box
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm,SparseOpticalFlowAlgorithm,SparseOpticalFlowConfigPreset
from metavision_sdk_core import BaseFrameGenerationAlgorithm
OUT=WORK/'output/local_displacement_20261026';assert (OUT/'mechanism_checks.json').exists()
assert not (OUT/'summary.json').exists()
cv2.setNumThreads(2);cv2.setRNGSeed(261026);results={}
(OUT/'source_snapshot').mkdir(exist_ok=True)
for n in ['live_local_flow.py','tools/check_local_flow.py','tools/probe_local_flow.py']:
    p=OUT/'source_snapshot'/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes((ROOT/n).read_bytes())
cases=dict(side=WORK/'output/single_model_optimized_RAW_20261026/side',train2=WORK/'output/single_model_optimized_RAW_20261026/train2',test3=WORK/'output/heldout_motion_RAW_20261026')
for case,run in cases.items():
    protocol=json.loads((run/'protocol.json').read_text('utf-8'))
    saved={round(r['t'],6):r for r in map(json.loads,(run/'tracking.jsonl').read_text('utf-8').splitlines())}
    noise=ActivityNoiseFilterAlgorithm(1280,720,10000);buffer=noise.get_empty_output_buffer()
    sparse=SparseOpticalFlowAlgorithm(1280,720,SparseOpticalFlowConfigPreset.FastObjects);flowbuffer=sparse.get_empty_output_buffer()
    feature=FeatureFlow();rows=[];native=events=roi_events=0;lastbody=None;lastseen=None;costs=[]
    begin=max(0.,protocol['start']-.1);image=np.empty((720,1280,3),np.uint8)
    for t,e in native_batches(protocol['raw'],begin,protocol['duration']+protocol['start']-begin,build_index=False):
        native+=1;events+=len(e);noise.process_events(e,buffer);filtered=buffer.numpy();row=saved.get(round(t,6))
        if row is not None and box(row['body']) is not None:lastbody=row['body'];lastseen=t
        prior=box(lastbody) if lastbody is not None and t-lastseen<=.16 else None
        if prior is not None:
            margin=.2*(prior[2:]-prior[:2]);lo=prior[:2]-margin;hi=prior[2:]+margin
            select=(filtered['x']>=lo[0])&(filtered['x']<=hi[0])&(filtered['y']>=lo[1])&(filtered['y']<=hi[1])
            roi=filtered[select]
        else:roi=filtered[:0]
        start=time.perf_counter();sparse.process_events(roi,flowbuffer);costs.append((time.perf_counter()-start)*1000);roi_events+=len(roi)
        if row is None:continue
        BaseFrameGenerationAlgorithm.generate_frame(filtered,image);sha=hashlib.sha256(image.tobytes()).hexdigest();assert sha==row['detector_frame_sha256']
        start=time.perf_counter();klt=feature.step(t,event_support(filtered),row['body']);klt_ms=(time.perf_counter()-start)*1000
        sdk=summarize_sparse(flowbuffer.numpy(),t,row['body'])
        rows.append(dict(t=t,frame_sha256=sha,klt=klt,sparse=sdk,klt_processing_ms=klt_ms))
    assert len(rows)==len(saved)
    summary=dict(matched_frames=len(rows),native_slices=native,native_events=events,roi_events=roi_events,
        qualified={n:sum(r[n]['valid'] for r in rows) for n in ['klt','sparse']},
        reasons={n:dict(Counter(r[n]['reason'] for r in rows)) for n in ['klt','sparse']},
        klt_median_ms=float(np.median([r['klt_processing_ms'] for r in rows])),sdk_per_native_median_ms=float(np.median(costs)),
        sdk_per_native_p95_ms=float(np.percentile(costs,95)))
    (OUT/(case+'.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8');results[case]=summary;print(json.dumps({case:summary}),flush=True)
(OUT/'summary.json').write_text(json.dumps(dict(results=results,all_native_frame_hashes_exact=True,labels_read=False,
    scope='Offline observer uses frozen current NN boxes; not RAW-only production or an end-to-end latency comparison.'),indent=2),encoding='utf-8')
