"""Serial actual RAW runs; fixed rule, existing detector, no cached input."""
from pathlib import Path
import sys,json,subprocess
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT.parent/'output/stage154_curved_rollout_20261024'
cases=[('clear','E:/dataset/train/6/media/event.raw',22.,8.),
    ('side','E:/dataset/train/6/media/event.raw',110.,9.5),
    ('train2','E:/dataset/train/2/aligned/event.raw',40.,6.)]
if len(sys.argv)>1:
    sys.path.insert(0,str(ROOT));from run_optimized_forecast import run
    name,raw,start,duration=next(c for c in cases if c[0]==sys.argv[1])
    run(raw,OUT/name,start,duration,True,True,pose_solver='lm',
        curved_rollout=True,skip_unobserved_geometry=True,preview_window_ms=5.)
else:
    for name,_,_,_ in cases:
        r=subprocess.run([sys.executable,str(Path(__file__).resolve()),name])
        if r.returncode:raise RuntimeError((name,r.returncode))
    print('THREE_RAW_RUNS_COMPLETE',flush=True)
