from pathlib import Path
import subprocess,sys
entry=Path(__file__).resolve().parents[1]/'research/stage67/study67.py'
def run(action,case=None):
    args=[sys.executable,str(entry),action]
    if case:args+=['--case',case]
    subprocess.run(args,check=True)
run('freeze')
for case in ['clear6','side6','train3','negative5']:run('run',case);run('audit',case)
