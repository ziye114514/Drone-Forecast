"""RAW inline author-IMM CV/physical branch interaction, retaining original33."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage73'),str(ROOT/'research/stage74')]
from online73 import OnlineExtras as PreviousExtras
from hybrid74 import HybridForecast
class OnlineExtras(PreviousExtras):
    def __init__(self,w,h):super().__init__(w,h);self.hybrid=HybridForecast()
    def forecast(self,q,obs,rotor_quality):
        q=super().forecast(q,obs,rotor_quality)
        return self.hybrid.step(q,obs,self.current['body'],rotor_quality)
