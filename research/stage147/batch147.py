"""Four frozen-configuration actual RAW runs, one GPU process at a time."""
from pathlib import Path
import sys,json,subprocess,argparse
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT.parent/'output/stage147_curvature_split_20261024'
cases=[('clear','E:/dataset/train/6/media/event.raw',22.,8.),
       ('side','E:/dataset/train/6/media/event.raw',110.,9.5),
       ('train2','E:/dataset/train/2/aligned/event.raw',40.,6.),
       ('fred0',str(ROOT.parent/'output/stage146_threshold_branches_20261024/input/fred0/events.raw'),108.,3.)]
p=argparse.ArgumentParser();p.add_argument('--suffix',default='');a=p.parse_args()
for name,raw,start,duration in cases:
    destination=OUT/(name+a.suffix)
    if (destination/'completion.json').exists():continue
    print(json.dumps(dict(starting=name)),flush=True)
    completed=subprocess.run([sys.executable,str(ROOT/'core/run_optimized_forecast.py'),'--raw',raw,'--output',str(destination),
        '--start',str(start),'--duration',str(duration),'--rotor-curvature','--preview'])
    if completed.returncode:raise SystemExit(completed.returncode)
