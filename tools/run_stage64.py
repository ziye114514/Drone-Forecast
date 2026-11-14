from pathlib import Path
import subprocess,sys
entry=Path(__file__).resolve().parents[1]/'research/stage64/study64.py'
def run(action,case=None):
    args=[sys.executable,str(entry),action]
    if case:args+=['--case',case]
    subprocess.run(args,check=True)
run('freeze');run('synthetic_run')
for c in ['clear6','side6','train3','negative5','prefix_side6','prefix_train3']:run('run',c)
run('prefix')
