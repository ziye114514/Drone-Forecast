"""Scoring-only path adapter; frozen inference/source protocol unchanged."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/stage22'))
import study69
if __name__=='__main__':study69.score()
