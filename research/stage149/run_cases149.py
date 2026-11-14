from pathlib import Path
import sys,json,time,traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from run_optimized_forecast import run
OUT=ROOT.parent/'output/stage149_inertia_scale_20261024'
cases=[('clear','E:/dataset/train/6/media/event.raw',22.,8.),
       ('side','E:/dataset/train/6/media/event.raw',110.,9.5),
       ('train2','E:/dataset/train/2/aligned/event.raw',40.,6.),
       ('fred0',str(ROOT.parent/'output/stage146_threshold_branches_20261024/input/fred0/events.raw'),108.,3.)]
# The audit hook intentionally persists and forbids reading old output caches.
# Each case therefore runs in a fresh interpreter, including local FRED RAW.
if len(sys.argv)>1:
    name,raw,start,duration=next(q for q in cases if q[0]==sys.argv[1])
    run(raw,OUT/name,start,duration,True,True,rotor_inertia=True,preview_window_ms=5.)
else:
    import subprocess,os
    for name,_,_,_ in cases:
        destination=OUT/name
        if destination.exists(): raise FileExistsError(destination)
        result=subprocess.run([sys.executable,str(Path(__file__)),name],creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode: raise RuntimeError((name,result.returncode))
    print('FOUR_RAW_CASES_FINISHED',flush=True)
