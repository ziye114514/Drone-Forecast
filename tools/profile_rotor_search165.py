"""Isolate current-input search cost; no trajectory labels or model changes."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,time,json,cProfile,pstats,io
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rotor_cuda_worker import CudaRotorFrontWorker
from rotor_support_front import estimate_region_translation


if __name__=='__main__':
    out=ROOT.parent/'output/native_plane165_20261104';maps=out/'native_maps';maps.mkdir(exist_ok=True)
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True)
    try:
        for clip in ['clear','side']:
            for index in [8,9,14,15]:
                path=maps/f'{clip}_{index:04d}.npz'
                if path.exists():continue
                with np.load(ROOT.parent/f'output/rotor_pipeline_20261101/cache/{clip}/{index:04d}.npz') as z:e=z['events'];t=float(z['t'])
                on,off,meta,_,_,_=worker.prepare_uncached(e,t)
                np.savez_compressed(path,on=on,off=off,native=meta['native_precomputed'][0],t=t)
    finally:worker.close()
    cv2.setNumThreads(2);records=[];prof=cProfile.Profile()
    for clip in ['clear','side']:
        for first in [8,14]:
            with np.load(maps/f'{clip}_{first:04d}.npz') as a,np.load(maps/f'{clip}_{first+1:04d}.npz') as b:
                args=((a['on'],a['off']),a['native'],(b['on'],b['off']));support=b['native']
            outputs={}
            for coarse in [False,True,True,False]:
                tick=time.perf_counter();cpu=time.process_time();prof.enable()
                shift=estimate_region_translation(*args,current_support=support,native_matching=True,subpixel=True,coarse_matching=coarse)
                prof.disable();records.append(dict(clip=clip,first=first,coarse=coarse,wall_ms=(time.perf_counter()-tick)*1000,cpu_ms=(time.process_time()-cpu)*1000,shift=shift))
                outputs[coarse]=shift
    stream=io.StringIO();pstats.Stats(prof,stream=stream).sort_stats('cumtime').print_stats(22)
    (out/'search_profile.txt').write_text(stream.getvalue(),'utf-8')
    (out/'search_profile.json').write_text(json.dumps(records,indent=2),'utf-8')
    print(stream.getvalue())
    print(json.dumps({str(v):dict(mean_ms=float(np.mean([r['wall_ms'] for r in records if r['coarse']==v])),mean_cpu_ms=float(np.mean([r['cpu_ms'] for r in records if r['coarse']==v]))) for v in [False,True]}))
