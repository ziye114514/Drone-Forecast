"""Existing per-rotor phase/FFT squared sum as a relative force input."""
from pathlib import Path
from collections import deque
import sys,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage88'),str(ROOT/'research/stage82'),str(ROOT/'research/stage72')]
from paired88 import Paired
from increment72 import ContinuousFrequency
from stream82 import StreamingActivity
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.rotor_sensing import PaperPerception,RotorSensing


class RotorActivity(StreamingActivity):
    def __init__(self,w,h):
        super().__init__(w,h);self.cfg99=PaperConfig();self.geometry99=PaperPerception(w,h,self.cfg99);self.sensing99=RotorSensing(self.cfg99);self.history99=deque()
    def step(self,e,bbox,t,velocity):
        value=super().step(e,bbox,t,velocity)
        crop=e[:0] if bbox is None else e[(e['x']>=bbox[0])&(e['x']<bbox[2])&(e['y']>=bbox[1])&(e['y']<bbox[3])]
        obs,_=self.geometry99.detect(crop,t);self.history99.append((t,e.copy()))
        while self.history99 and self.history99[0][0]<t-self.cfg99.frequency_window_us/1e6-.000001:self.history99.popleft()
        measurement=self.sensing99.step(obs,self.history99,t)
        all_current=bool(len(self.sensing99.tracks)==4 and all(q.frequency_t is not None and abs(q.frequency_t-t)<1e-8 and abs(q.seen-t)<1e-8 for q in self.sensing99.tracks))
        value['rotor99']=dict(measurement,all_four_measured_current=all_current)
        return value


class RotorFrequency(ContinuousFrequency):
    def step(self,t,activity):
        r=activity['rotor99'];candidates=[]
        if r['four_frequencies'] and r['all_four_measured_current']:
            hz=float(np.sqrt(r['sum_frequency_squared']/4.))
            candidates=[dict(qualified=True,frequency_hz=hz,tone_fraction=max(r['frequency_quality'],1e-9),phase_coherence=1.)]
        value,d=super().step(t,dict(spectral_candidates=candidates))
        d.update(measurement_source='four_rotor_phaseFFT_sum_squared',all_four_measured_current=r['all_four_measured_current'],
            four_frequencies=r['four_frequencies'],sum_frequency_squared=r['sum_frequency_squared'],
            rotor_hz=[x['blade_pass_hz'] for x in r['rotors']],rotor_phase=[x['frequency_diagnostics'].get('phase_coherence') for x in r['rotors']],
            continuity_weight='Mean rotor quality times neutral1, not a new phase measurement.',mechanical_RPM_verified=False)
        return value,d


class RotorPaired(Paired):
    def reset(self):super().reset();self.inc.frequency=RotorFrequency()
