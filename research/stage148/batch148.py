"""Frozen148 configuration, serial actual RAW-only inference."""
from pathlib import Path
import subprocess,sys,json,argparse
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT.parent/'output/stage148_temporal_rotors_20261024'
p=argparse.ArgumentParser();p.add_argument('--suffix',default='');a=p.parse_args()
for name,raw,start,duration in [
    ('clear','E:/dataset/train/6/media/event.raw',22.,8.),
    ('side','E:/dataset/train/6/media/event.raw',110.,9.5),
    ('train2','E:/dataset/train/2/aligned/event.raw',40.,6.),
    ('fred0',str(ROOT.parent/'output/stage146_threshold_branches_20261024/input/fred0/events.raw'),108.,3.)]:
    out=OUT/(name+a.suffix)
    if (out/'completion.json').exists():continue
    print(json.dumps(dict(starting=name)),flush=True)
    result=subprocess.run([sys.executable,str(ROOT/'run_optimized_forecast.py'),'--raw',raw,'--output',str(out),
                           '--start',str(start),'--duration',str(duration),'--rotor-temporal','--preview'])
    if result.returncode:raise SystemExit(result.returncode)
