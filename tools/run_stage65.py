from pathlib import Path
import subprocess,sys
entry=Path(__file__).resolve().parents[1]/'research/stage65/study65.py'
for action in ['freeze','infer','prefix','score']:subprocess.run([sys.executable,str(entry),action],check=True)
