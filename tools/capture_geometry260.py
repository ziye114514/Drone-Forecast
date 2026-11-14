"""Capture EXACT original masked geometry counts, not a new detector.

259 raw-count control did not match because the front masks20ms counts with
period-seeded native support. Preserve that mask and every selected timestamp
before isolating geometry motion compensation. No heldout recording is read.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,argparse,hashlib,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from run_rotor_only import RotorPredictor,serial
from rotor_cuda_worker import CudaRotorFrontWorker
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from native_history_buffer import NativeHistoryBuffer
from fast_structure_front import roi_bounds,count_pair
from pure_visibility170 import window
from replay_joint236 import source

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',choices=['13','15','16'],required=True);a=p.parse_args()
    OUT=ROOT.parent/f'output/geometry260_20261108/recording{a.recording}';OUT.mkdir(parents=True,exist_ok=False)
    files=['tools/capture_geometry260.py','fast_structure_front.py','rotor_only_front.py','native_event_alignment.py']
    for n in files:
        dst=OUT/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/n).read_bytes())
    (OUT/'protocol.json').write_text(json.dumps(dict(recording=a.recording,warp_enabled=False,NN_used=False,
        thresholds_changed=False,RAW_FPS_measured=False,heldout_opened=False,
        purpose='exact original spatial mask/count/body capture before any geometry intervention',
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}),indent=2),'utf-8')
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    model=RotorPredictor(no_phase=True,fast_geometry=True,local_phase=True,subpixel_motion=True,entry_init=True,
        relative_observation=True,relock_support_guard=True,border_motion=True,gap_prediction='short',reference_mode='fixed',unknown_force_model='plane-adaptive',projected_fov_guard=True)
    model.front.current_region_search=True;model.period_worker=worker;model.front.period_estimator=worker;model.front.history_buffer=NativeHistoryBuffer()
    geometry=model.front.geometry;original=geometry.step;captured={}
    def capture(events,t,body,q,velocity,**kwargs):
        captured.clear();captured.update(t=t,body=copy.deepcopy(body),velocity=np.asarray(velocity).tolist(),kwargs=kwargs)
        if body.get('detected'):
            bounds=roi_bounds(body['bbox']);on,off=geometry.count_current(events,bounds,t,geometry.selector)
            captured.update(bounds=bounds.tolist(),on=on.copy(),off=off.copy())
        return original(events,t,body,q,velocity,**kwargs)
    geometry.step=capture
    rows=[json.loads(s) for s in source(a.recording).read_text('utf-8').splitlines()]
    cache=ROOT.parent/f'output/{"motion181_20261107" if a.recording=="15" else "phase185_20261107"}/cache_recording{a.recording}'
    metadata=[];maxdelta=0.;counts_delta=0.
    try:
        for i,old in enumerate(rows):
            t=old['t'];events=window(cache,t);prepared=worker.prepare_uncached(events,t);worker.cached=(t,prepared)
            r,frame=model.step(events[:0],t)
            if r['coefficient']:
                for h,pred in r['physical'].items():maxdelta=max(maxdelta,float(np.max(abs(np.asarray(pred)-old['physical'][h]))))
            item={k:v for k,v in captured.items() if k not in ('on','off')}
            item.update(index=i,source_structure=r['structure'],source_geometry=r['geometry'],post_body=r['body'])
            if captured['body'].get('detected'):
                mask=frame['geometry_native_mask'];keep=(events['t']>=round(t*1e6)-20000)&mask[events['y'],events['x']]
                selected=events[keep].copy();on,off=count_pair(selected,np.asarray(captured['bounds']),t,cpp_histogram=True)
                counts_delta=max(counts_delta,float(np.max(abs(on-captured['on']))),float(np.max(abs(off-captured['off']))))
                assert counts_delta==0.,counts_delta
                name=f'input_{i:04d}.npz';np.savez_compressed(OUT/name,on=captured['on'],off=captured['off'],events=selected);item['file']=name
            metadata.append(item)
            if i%30==0:print(a.recording,i,'forecast max delta',maxdelta,'count delta',counts_delta,flush=True)
            (OUT/'manifest.partial.json').write_text(json.dumps(metadata,default=serial),'utf-8')
        assert maxdelta<1e-6,maxdelta
    finally:
        model.front.previous=None;model.front.geometry.count_current=None;model.period_worker=None;worker.cached=None;worker.close();configure_native(None)
    (OUT/'manifest.json').write_text(json.dumps(dict(recording=a.recording,records=metadata,original_forecast_max_px=maxdelta,
        masked_native_counts_max_delta=counts_delta,NN_used=False,warp_enabled=False),default=serial,indent=2),'utf-8')
    print(json.dumps(dict(recording=a.recording,frames=len(rows),original_forecast_max_px=maxdelta,masked_native_counts_max_delta=counts_delta)),flush=True)
