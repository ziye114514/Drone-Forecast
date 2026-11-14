"""Fresh processes keep the inference truth-access guard scoped to inference."""
from pathlib import Path
import subprocess, sys
entry=Path(__file__).resolve().parents[1]/'research/stage62/study62.py'
for action in ['freeze','infer','prefix','score']:
    subprocess.run([sys.executable,str(entry),action],check=True)
