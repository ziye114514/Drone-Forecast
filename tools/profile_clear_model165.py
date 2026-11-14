"""Profile only the CPU estimator after exact current GPU features are ready."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,cProfile,pstats,io,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
import numpy as np,cv2
from run_rotor_only import RotorPredictor,serial

if __name__=='__main__':
    cv2.setNumThreads(2)
    model=RotorPredictor('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',gpu_front=True,native_gpu=True,
        fast_geometry=True,no_phase=True,local_phase=True,subpixel_motion=True,entry_init=True,remote_front=True,
        relative_observation=True,relock_support_guard=True,border_motion=True)
    profile=cProfile.Profile();records=[]
    try:
        for index in range(24):
            with np.load(ROOT.parent/f'output/rotor_pipeline_20261101/cache/clear/{index:04d}.npz') as z:e=z['events'].copy();t=float(z['t'])
            result=model.period_worker.prepare_uncached(e,t);model.period_worker.cached=(t,result)
            profile.enable();r,frame=model.step(e[:0],t);profile.disable();records.append(r)
    finally:
        frame=None;model.front.previous=None;model.front.geometry.count_current=None;model.period_worker.close()
    output=ROOT.parent/'output/native_plane165_20261104'
    s=io.StringIO();pstats.Stats(profile,stream=s).sort_stats('cumtime').print_stats(35)
    (output/'clear_model_profile.txt').write_text(s.getvalue(),'utf-8')
    (output/'clear_profile_rows.jsonl').write_text(''.join(json.dumps(r,default=serial)+'\n' for r in records),'utf-8');print(s.getvalue())
