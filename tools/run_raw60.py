from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/stage60'))
import raw60
raw60.freeze();raw60.run()
