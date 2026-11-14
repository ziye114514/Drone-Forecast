from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'research/stage59'))
import study59
for action in ['freeze','synthetic','real','report']:
    print('Stage59 '+action,flush=True)
    getattr(study59,action)()
