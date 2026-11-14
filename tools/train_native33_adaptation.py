"""Reuse the existing guarded trainer; fixed native33 data and5 epochs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import train_fred_adaptation as training


if __name__=='__main__':
    training.OUT=Path(sys.argv[1])/'native33_adaptation'
    training.main()
