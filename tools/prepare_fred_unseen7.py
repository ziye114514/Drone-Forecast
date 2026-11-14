"""Reuse the frozen interval-selection rule on an adaptation-excluded recording."""
from pathlib import Path
import prepare_fred_transfer as preparation
preparation.OUT=preparation.OUT/'unseen7'
preparation.OUT.parent.mkdir(exist_ok=True)
preparation.SEQUENCES=(7,)
if __name__=='__main__':preparation.main()
