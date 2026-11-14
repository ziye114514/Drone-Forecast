from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/stage59'))
import lsd59
for action in ['freeze','run','report']:getattr(lsd59,action)()
