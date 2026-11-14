"""Same exposed observations/endpoints, one visibility-based refinement."""
from pathlib import Path
import json,sys,types
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage140'),str(ROOT/'research/stage139')]
import study139
from visible140 import VisibleGate,PARAMS
OUT=ROOT.parent/'output/stage140_visible_fallback_20261024'
if __name__=='__main__':
    namespace=dict(study139.__dict__,OUT=OUT,DenseGate=VisibleGate,PARAMS=PARAMS)
    types.FunctionType(study139.main.__code__,namespace)()
