"""One native decode, spatial diagnostics, and a reusable exact support cache."""
from pathlib import Path
import hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path.insert(0,str(ROOT))
from live_decode import native_batches
from live_local_flow import FeatureFlow,frame_support,event_support
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_core import BaseFrameGenerationAlgorithm as Frame
OUT=WORK/'output/spatial_flow_20261026';assert (OUT/'mechanism_checks.json').exists() and not (OUT/'summary.json').exists()
cv2.setNumThreads(2);cv2.setRNGSeed(261026);summary={}
cases=dict(side=WORK/'output/local_displacement_tight_20261026/actual_side',
    train2=WORK/'output/local_displacement_tight_20261026/actual_train2',test3=WORK/'output/heldout_motion_RAW_20261026')
for case,run in cases.items():
    protocol=json.loads((run/'protocol.json').read_text('utf-8'))
    saved={round(r['t'],6):r for r in map(json.loads,(run/'tracking.jsonl').read_text('utf-8').splitlines())}
    reference={round(r['t'],6):r['klt'] for r in json.loads((WORK/'output/local_displacement_tight_20261026'/(case+'.json')).read_text('utf-8'))} if case=='test3' else None
    noise=ActivityNoiseFilterAlgorithm(1280,720,10000);buffer=noise.get_empty_output_buffer()
    observer=FeatureFlow(crop=True,padding=112,spatial_diagnostics=True)
    image=np.empty((720,1280,3),np.uint8);rows=[];cache=[];metadata=[];deltas=[];mismatches=[];timings=[]
    begin=max(0.,protocol['start']-.1)
    for t,e in native_batches(protocol['raw'],begin,protocol['duration']+protocol['start']-begin,build_index=False):
        noise.process_events(e,buffer);filtered=buffer.numpy();old=saved.get(round(t,6))
        if old is None:continue
        Frame.generate_frame(filtered,image);sha=hashlib.sha256(image.tobytes()).hexdigest();assert sha==old['detector_frame_sha256']
        support=frame_support(image,Frame.bg_color_default()[0]);assert np.array_equal(support,event_support(filtered))
        packed=np.packbits(support.ravel()>0);cache.append(packed)
        metadata.append(dict(t=t,body=old['body'],frame_sha256=sha,support_sha256=hashlib.sha256(support.tobytes()).hexdigest()))
        start=time.perf_counter();observation=observer.step(t,support,old['body']);timings.append((time.perf_counter()-start)*1000)
        prior=reference[round(t,6)] if reference is not None else old['diagnostic']['local_flow']['observation']
        if observation['valid']!=prior['valid']:mismatches.append(t)
        if observation['valid'] and prior['valid']:deltas.append(float(np.linalg.norm(np.asarray(observation['displacement'])-prior['displacement'])))
        rows.append(dict(t=t,frame_sha256=sha,observation=observation))
    assert len(rows)==len(saved)
    assert not mismatches and max(deltas,default=0.)<.001,dict(case=case,mismatches=mismatches,max_displacement_delta=max(deltas,default=0.))
    np.savez_compressed(OUT/(case+'_support.npz'),packbits=np.stack(cache),shape=np.array([720,1280],dtype=np.int32))
    (OUT/(case+'_support.json')).write_text(json.dumps(dict(scope='Offline spatial-only native support cache; not for RPM or production inference.',
        original_run=str(run),raw=protocol['raw'],records=metadata),indent=2),encoding='utf-8')
    (OUT/(case+'.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
    qualified=[r['observation']['spatial_support'] for r in rows if r['observation']['valid']]
    summary[case]=dict(frames=len(rows),v4_qualified=len(qualified),center_in_hull=sum(s['center_in_hull'] for s in qualified),
        enough_spatial_fits=sum(s['extra_velocity_covariance'] is not None for s in qualified),
        max_v4_displacement_delta_px=max(deltas,default=0.),median_processing_with_spatial_ms=float(np.median(timings)),
        support_cache_bytes=(OUT/(case+'_support.npz')).stat().st_size,frame_hashes_and_support_exact=True)
    print(json.dumps({case:summary[case]}),flush=True)
(OUT/'summary.json').write_text(json.dumps(dict(results=summary,labels_read=False,current_observation_only=True,
    no_new_NN_inference=True,geometry_qualifies_velocity_only=True),indent=2),encoding='utf-8')
