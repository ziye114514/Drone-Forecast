"""Serial actual-RAW runs; one GPU worker at a time, no scoring in this process."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent/'output/stage146_threshold_branches_20261024'
CASES = [
    ('clear4', 'E:/dataset/train/6/media/event.raw', 22., 8.),
    ('side4', 'E:/dataset/train/6/media/event.raw', 110., 9.5),
    ('train2_4', 'E:/dataset/train/2/aligned/event.raw', 40., 6.),
    ('fred0_4', str(OUT/'input/fred0/events.raw'), 108., 3.),
]
parser = argparse.ArgumentParser(); parser.add_argument('--suffix', default='')
suffix = parser.parse_args().suffix

for name, raw, start, duration in CASES:
    destination = OUT/(name+suffix)
    if (destination/'completion.json').exists():
        print(json.dumps(dict(case=name, already_completed=True)), flush=True)
        continue
    print(json.dumps(dict(case=name, status='starting', raw=raw, start=start, duration=duration)), flush=True)
    result = subprocess.run([sys.executable, str(ROOT/'core/run_optimized_forecast.py'), '--raw', raw,
                             '--output', str(destination), '--start', str(start), '--duration', str(duration),
                             '--rotor-branches', '--rotor-threshold', '4', '--preview'])
    if result.returncode:
        raise SystemExit(result.returncode)
    print(json.dumps(dict(case=name, status='completed')), flush=True)
