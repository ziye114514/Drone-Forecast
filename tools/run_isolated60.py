from pathlib import Path
import subprocess,sys
entry=Path(__file__).resolve().parents[1]/'research/stage60/isolated60.py'
subprocess.run([sys.executable,str(entry),'freeze'],check=True)
for case in ['clear6','side6','train3','negative5']:subprocess.run([sys.executable,str(entry),'run','--case',case],check=True)
subprocess.run([sys.executable,str(entry),'summary'],check=True)
