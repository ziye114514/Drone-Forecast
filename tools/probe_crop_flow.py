"""Aligned padded crop speedup; compare recorded full-frame flow on same RAW."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path.insert(0,str(ROOT))
from live_decode import native_batches
from live_local_flow import FeatureFlow,event_support,frame_support
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_core import BaseFrameGenerationAlgorithm as Frame
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path);parser.add_argument('--padding',type=int,default=192);args=parser.parse_args()
OUT=args.output_dir.resolve() if args.output_dir else WORK/'output/local_displacement_crop_20261026';REFERENCE=WORK/'output/local_displacement_20261026'
assert (OUT/'mechanism_checks.json').exists() and not (OUT/'summary.json').exists()
cv2.setNumThreads(2);cv2.setRNGSeed(261026);results={}
cases=dict(side=WORK/'output/single_model_optimized_RAW_20261026/side',train2=WORK/'output/single_model_optimized_RAW_20261026/train2',test3=WORK/'output/heldout_motion_RAW_20261026')
for case,run in cases.items():
    protocol=json.loads((run/'protocol.json').read_text('utf-8'))
    saved={round(r['t'],6):r for r in map(json.loads,(run/'tracking.jsonl').read_text('utf-8').splitlines())}
    old={round(r['t'],6):r for r in json.loads((REFERENCE/(case+'.json')).read_text('utf-8'))}
    noise=ActivityNoiseFilterAlgorithm(1280,720,10000);buffer=noise.get_empty_output_buffer();feature=FeatureFlow(crop=True,padding=args.padding)
    begin=max(0.,protocol['start']-.1);image=np.empty((720,1280,3),np.uint8);rows=[];errors=[];mismatches=[]
    for t,e in native_batches(protocol['raw'],begin,protocol['duration']+protocol['start']-begin,build_index=False):
        noise.process_events(e,buffer);filtered=buffer.numpy();row=saved.get(round(t,6))
        if row is None:continue
        Frame.generate_frame(filtered,image);sha=hashlib.sha256(image.tobytes()).hexdigest();assert sha==row['detector_frame_sha256']
        start=time.perf_counter();support=frame_support(image,Frame.bg_color_default()[0]);result=feature.step(t,support,row['body']);elapsed=(time.perf_counter()-start)*1000
        assert np.array_equal(support,event_support(filtered)), 'Frame support changed ON/OFF event pixels'
        prior=old[round(t,6)];assert sha==prior['frame_sha256']
        if result['valid']!=prior['klt']['valid'] or result['reason']!=prior['klt']['reason']:mismatches.append(t)
        if result['valid'] and prior['klt']['valid']:errors.append(float(np.linalg.norm(np.asarray(result['displacement'])-prior['klt']['displacement'])))
        rows.append(dict(t=t,frame_sha256=sha,klt=result,sparse=prior['sparse'],sparse_reused_from_full_probe=True,klt_processing_ms=elapsed))
    assert len(rows)==len(saved)
    (OUT/(case+'.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
    results[case]=dict(matched_frames=len(rows),qualified=sum(r['klt']['valid'] for r in rows),acceptance_mismatch_times=mismatches,
        max_displacement_difference_px=max(errors,default=0.),klt_median_ms=float(np.median([r['klt_processing_ms'] for r in rows])),
        frame_support_matches_filtered_native_events=True)
    print(json.dumps({case:results[case]}),flush=True)
(OUT/'summary.json').write_text(json.dumps(dict(results=results,aligned_padding_px=args.padding,pyramid_alignment_px=8,
    current_frame_hashes_exact=True,sparse_observer_reused=True,labels_read=False),indent=2),encoding='utf-8')
