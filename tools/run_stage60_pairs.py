from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/stage60'))
import study60
for action in ['freeze','run','report']:getattr(study60,action)()
