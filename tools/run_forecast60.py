"""Separate processes preserve original inference audit guards between runs."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1];entry=ROOT/'research/stage60/forecast60.py'
def run(action,case=None):
    args=[sys.executable,str(entry),action]
    if case:args+=['--case',case]
    subprocess.run(args,check=True,cwd=ROOT)
run('freeze')
for case in ['pilot_off_clear6','clear6','side6','train3','negative5','prefix_side6']:
    run('run',case);run('audit',case)
run('prefix');run('score')
