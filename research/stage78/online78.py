"""Actual RAW integration of77 structural fixes; original39 retained."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage75'),str(ROOT/'research/stage77')]
from online75 import OnlineExtras as PreviousExtras
from separated77 import SeparatedForecast
class OnlineExtras(PreviousExtras):
    def __init__(self,w,h):super().__init__(w,h);self.separated=SeparatedForecast()
    def forecast(self,q,obs,rotor_quality):
        q=super().forecast(q,obs,rotor_quality)
        return self.separated.step(q,obs)
