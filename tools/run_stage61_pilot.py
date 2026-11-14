from pathlib import Path
import subprocess,sys
entry=Path(__file__).resolve().parents[1]/'research/stage61/study61.py'
subprocess.run([sys.executable,str(entry),'freeze'],check=True)
for mode in ['normal','consume_rng']:
    for case in ['clear6','side6','train3','negative5']:subprocess.run([sys.executable,str(entry),'run','--mode',mode,'--case',case],check=True)
subprocess.run([sys.executable,str(entry),'report'],check=True)
