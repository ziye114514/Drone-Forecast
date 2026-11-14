from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1];entry=ROOT/'research/stage60/forecast60.py';checks=entry.with_name('checks60.py')
def run(path,action,case=None):
    args=[sys.executable,str(path),action]
    if case:args+=['--case',case]
    subprocess.run(args,check=True,cwd=ROOT)
run(checks,'freeze');run(checks,'audit','pilot_off_clear6')
for case in ['clear6','side6','train3','negative5','prefix_side6']:
    run(entry,'run',case);run(checks,'audit',case)
run(entry,'prefix');run(entry,'score')
