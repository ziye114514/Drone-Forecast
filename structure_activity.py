"""Object-level current periodicity independent of individual motor labels.

Median blade-passage frequency is a motion feature, not the sum of four shaft
rates or a Newton thrust measurement. No hidden rotor or held period is added.
"""
from collections import deque
import numpy as np


class WholeBodyPeriodicity:
    def __init__(self):self.epoch=None;self.history=deque();self.reference=None
    def step(self,t,rates,epoch):
        if self.epoch!=epoch:self.epoch=epoch;self.history.clear();self.reference=None
        frequencies=[];pixels=[]
        for rate in rates:
            current=rate.get('instantaneous146',rate)
            if rate.get('past_only_frequency_held') or not current.get('valid'):continue
            f=current.get('frequency_hz')
            if f is None or not np.isfinite(f) or f<=0:continue
            if current.get('pixels',0)<8 or current.get('mode_fraction',0.)<.35:continue
            frequencies.append(float(f));pixels.append(int(current['pixels']))
        while self.history and t-self.history[0][0]>.08:self.history.popleft()
        out=dict(valid=False,reason='few_current_periodic_regions',current_regions=len(frequencies),
            current_frequencies_hz=frequencies,physical_motor_identity_required=False,
            mechanical_RPM_measured=False,total_thrust_N=None,hidden_periods_added=0)
        if len(frequencies)<2:return out
        frequency=float(np.median(frequencies))
        coherent=[q for q in self.history if abs(np.log(frequency/q[1]))<=.08]
        self.history.append((t,frequency))
        if not coherent:return dict(out,reason='waiting_for_object_period_consistency')
        if self.reference is None:self.reference=frequency
        return dict(out,valid=True,reason='current_object_periodic_feature',
            frequency_hz=frequency,reference_hz=self.reference,
            relative_f2_proxy=float((frequency/self.reference)**2),
            identity='whole detected object; not individual motors',
            harmonic_identity_unverified=True,coherent_frames=len(coherent)+1)
