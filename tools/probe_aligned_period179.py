"""Reuse the existing CUDA temporal test on raw and past-only aligned events.

Native176 past velocity is frozen before each complete window. These are
mechanism probes, never new RPM truth or a forecast/FPS claim.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_cuda_worker import CudaRotorFrontWorker
from native_event_alignment import past_warp
from pure_visibility170 import window
from curvature_visibility import masks,contours
from run_rotor_only import serial
D=ROOT.parent/'output/rim174_20261107';OUT=ROOT.parent/'output/alignment178_20261107'

if __name__=='__main__':
    cv2.setNumThreads(2);out=[]
    rows=[json.loads(s) for s in (ROOT.parent/'output/layout176_20261107/integrated/train2_fixed.jsonl').read_text('utf-8').splitlines()]
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,regional_seeds=True,remote_unpack=True)
    prepared=None
    try:
        for rec in json.loads((D/'results.json').read_text('utf-8')):
            t=rec['t'];q=round(t,1);old=max((r for r in rows if r['t']<=t-.033333),key=lambda r:r['t']);m=old['motion'];s=m.get('estimate')
            if s is None or s.get('peak_at_search_boundary') or not old['body'].get('detected'):raise RuntimeError('Missing past native velocity')
            velocity=np.array([s['dx'],s['dy']])/m['dt_s']
            events=window(ROOT.parent/'output/native_plane165_20261104/cache_train2_inside',t)
            warped,provenance=past_warp(events,t,velocity,old['t'])
            z=np.load(D/f'{q:.1f}.npz');x,y,X,Y=rec['roi'];keep=z['geometry_native'][y:Y,x:X]
            a=np.where(keep,z['on'][y:Y,x:X],0);b=np.where(keep,z['off'][y:Y,x:X],0);_,outline=masks(a,b)
            arms={}
            for arm,e in [('raw',events),('past_aligned',warped)]:
                prepared=worker.prepare_uncached(e,t);period=prepared[-1];mask=period['mask']>0;frequency=period['frequency_hz'];parents=[]
                for parent in contours(outline)[:3]:
                    p=parent.copy();p[:,:,0]+=x;p[:,:,1]+=y
                    if arm=='past_aligned':p+=np.rint(.01*velocity).astype(np.int32)
                    select=np.zeros(mask.shape,np.uint8);cv2.drawContours(select,[p],-1,1,-1)
                    fs=frequency[(select>0)&mask];fs=fs[np.isfinite(fs)&(fs>=60)&(fs<=1200)]
                    mode_fraction=None;hz=None
                    if len(fs):
                        bins=np.floor(np.log(fs/60.)/np.log(1.04)).astype(int);hist=np.bincount(np.maximum(bins,0));peak=int(np.argmax(hist));chosen=abs(bins-peak)<=1;mode_fraction=float(chosen.mean());hz=float(np.median(fs[chosen]))
                    parents.append(dict(unique_period_pixels=len(fs),mode_fraction=mode_fraction,frequency_hz=hz))
                arms[arm]=dict(accepted_frame_pixels=period['accepted_pixels'],parents=parents,
                    parameters=period['parameters'])
                np.savez_compressed(OUT/f'{q:.1f}_{arm}_period179.npz',mask=mask,frequency_hz=frequency)
            out.append(dict(t=t,alignment=provenance,arms=arms));print(q,[(k,[r['unique_period_pixels'] for r in v['parents']]) for k,v in arms.items()],flush=True)
    finally:prepared=None;worker.cached=None;worker.close()
    (OUT/'aligned_period179.json').write_text(json.dumps(dict(records=out,labels_used=False,parameters_searched=False,forecast_or_RPM_truth=False),default=serial,indent=2),'utf-8')
