"""Each inference invocation has a fresh audit guard and independent worker lifecycle."""
from pathlib import Path
import subprocess,sys
entry=Path(__file__).resolve().parents[1]/'research/stage61/forecast61.py'
def run(action,case=None):
    args=[sys.executable,str(entry),action]
    if case:args+=['--case',case]
    subprocess.run(args,check=True)
run('freeze')
cases=['pilot_off_clear6']+[f'{mode}_{c}' for c in ['clear6','side6','train3','negative5'] for mode in ['off','paired']]+['prefix_off_side6','prefix_paired_side6','prefix_off_train3','prefix_paired_train3']
for case in cases:run('run',case);run('audit',case)
run('prefix');run('score')
