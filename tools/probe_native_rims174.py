"""Isolate the existing two-arcs skip on exact native20ms support.

Three current-frame hypotheses, not a forecast improvement/AP experiment.
No reference coordinates are read. Warm current candidates are frozen172
native-only outputs; current source pixels/period support come from CUDA.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_cuda_worker import CudaRotorFrontWorker
from rotor_only_front import RotorOnlyFront
from pure_visibility170 import window
from curvature_visibility import recover,masks,contours
from run_rotor_only import serial
OUT=ROOT.parent/'output/rim174_20261107'

if __name__=='__main__':
    OUT.mkdir(exist_ok=False);cv2.setNumThreads(2)
    rows=[json.loads(r) for r in (ROOT.parent/'output/geometry170_20261106/final_regional172/train2_fixed.jsonl').read_text('utf-8').splitlines()]
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,regional_seeds=True,remote_unpack=True)
    results=[];frame=None;prepared=None
    try:
        for q in (43.4,43.6,44.0):
            old=min(rows,key=lambda r:abs(r['t']-q));t=old['t'];events=window(ROOT.parent/'output/native_plane165_20261104/cache_train2_inside',t)
            prepared=worker.prepare_uncached(events,t);worker.cached=(t,prepared)
            f=RotorOnlyFront(worker,fast_geometry=True);f.provide_phase_events=False;frame=f.step(events[:0],t)
            on,off=frame['native_counts20'];keep=frame['geometry_native_mask'];x,y,X,Y=old['geometry']['roi']
            a=np.where(keep[y:Y,x:X],on[y:Y,x:X],0.);b=np.where(keep[y:Y,x:X],off[y:Y,x:X],0.)
            original=old['geometry'].get('rotors',[]);control,ca=recover(a,b,original,np.array([x,y]));forced,fa=recover(a,b,[],np.array([x,y]))
            rec=dict(t=t,roi=[x,y,X,Y],old=original,control=control,control_audit=ca,
                force_reconsider_parent_hypotheses=forced,force_audit=fa,NN_used=False,coordinate_labels_used=False,
                candidate_count_is_not_true_rotor_count=True)
            np.savez_compressed(OUT/f'{q:.1f}.npz',on=on,off=off,geometry_native=keep)
            results.append(rec);print(q,len(original),len(control),len(forced),fa['accepted_pairs'],flush=True)
            f.previous=None;f.geometry.count_current=None;frame=None;f=None
    finally:prepared=None;worker.cached=None;worker.close()
    (OUT/'results.json').write_text(json.dumps(results,default=serial,indent=2),'utf-8')
