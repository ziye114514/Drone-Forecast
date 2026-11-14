"""Measure actual current period votes on the existing native outline proposals."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_cuda_worker import CudaRotorFrontWorker
from pure_visibility170 import window
from curvature_visibility import masks,contours
from batched_arc_fit import fit_arc_exact
from run_rotor_only import serial
D=ROOT.parent/'output/rim174_20261107';OUT=ROOT.parent/'output/alignment178_20261107'

if __name__=='__main__':
    cv2.setNumThreads(2);out=[]
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,regional_seeds=True,remote_unpack=True)
    prepared=None
    try:
        for rec in json.loads((D/'results.json').read_text('utf-8')):
            t=rec['t'];q=round(t,1);events=window(ROOT.parent/'output/native_plane165_20261104/cache_train2_inside',t)
            prepared=worker.prepare_uncached(events,t);period=prepared[-1];strict=period['mask'];frequency=period['frequency_hz']
            z=np.load(D/f'{q:.1f}.npz');x,y,X,Y=rec['roi'];keep=z['geometry_native'][y:Y,x:X]
            on=np.where(keep,z['on'][y:Y,x:X],0);off=np.where(keep,z['off'][y:Y,x:X],0);_,outline=masks(on,off)
            records=[]
            for parent in contours(outline)[:3]:
                p=parent.reshape(-1,2);fit=fit_arc_exact(p,float(np.ptp(p,axis=0).max()))
                part=np.zeros_like(outline);cv2.drawContours(part,[parent],-1,1,-1)
                eligible=(part>0)&(strict[y:Y,x:X]>0)
                fs=frequency[y:Y,x:X][eligible];fs=fs[np.isfinite(fs)&(fs>=60)&(fs<=1200)]
                mode_fraction=None
                if len(fs):
                    bins=np.floor(np.log(fs/60.)/np.log(1.04)).astype(int);hist=np.bincount(np.maximum(bins,0));peak=int(np.argmax(hist));mode_fraction=float(np.mean(abs(bins-peak)<=1))
                records.append(dict(bbox=cv2.boundingRect(parent),unique_period_pixels=len(fs),mode_fraction=mode_fraction,fit=None if fit is None else {k:v for k,v in fit.items() if k!='points'}))
            np.savez_compressed(OUT/f'{q:.1f}_period.npz',mask=strict,frequency_hz=frequency)
            out.append(dict(t=t,parents=records));print(q,[(r['bbox'],r['unique_period_pixels'],r['mode_fraction']) for r in records],flush=True)
    finally:prepared=None;worker.cached=None;worker.close()
    (OUT/'period_seeds.json').write_text(json.dumps(out,default=serial,indent=2),'utf-8')
