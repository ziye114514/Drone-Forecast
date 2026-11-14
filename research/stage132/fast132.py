"""Reuse robust23 acceleration with the current fast107 state/rotor motion.

Do not feed an uncertain acceleration back into the position state. Estimate
it robustly from past body positions, shrink by its estimation variance and
use it only for the future rollout. Same acceleration for both controls.
"""
from pathlib import Path
from collections import deque
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage130'),str(ROOT/'research/stage23')]
from motion130 import RelativeVelocity,SimpleKF
from robust23 import robust_acceleration,cap,PARAMS as ROBUST
PARAMS=dict(acceleration='Actual unchanged23 robust_acceleration/cap; .35s history,6 points/.16s, Huber1.345*6px, error-based shrink. Keep fast107 state untouched by uncertain acceleration.',
    models=['fast_body_acc','fast_geometry_acc'],isolation='Same robust acceleration in both; only velocity observation differs. No Singer internal acceleration feedback.',
    window_s=.35,body_gap_s=.5,limits='Image-space heuristic, all clips previously exposed. No true force/COM calibration.')
def forecast(c,h):return (np.array(c['position'])+h*np.array(c['velocity'])+.5*h*h*np.array(c['acceleration'])).tolist() if c else None
class Fast:
    def __init__(self):self.body=SimpleKF();self.geometry=RelativeVelocity();self.history=deque();self.seen=None
    def step(self,t,body,models):
        if self.seen is not None and t-self.seen>.5:self.__init__()
        z=body['center'] if body.get('detected') else None
        if z is not None:self.history.append((t,np.asarray(z)));self.seen=t
        while self.history and t-self.history[0][0]>.35+1e-8:self.history.popleft()
        a,se,diag=robust_acceleration(self.history)
        shrink=np.maximum(0.,1.-se*se/(a*a+1e-12)) if diag['valid'] else np.zeros(2);a=cap(a*shrink)
        c=self.body.step(t,z);g=self.geometry.step(t,body,models);coeff={}
        for name,value in [('fast_body_acc',c),('fast_geometry_acc',g['coefficient'])]:
            value=copy.deepcopy(value)
            if value is not None:value['acceleration']=a.tolist()
            coeff[name]=dict(coefficient=value,source=g['source'] if name=='fast_geometry_acc' else None)
        return dict(coeff,acceleration_diagnostic=dict(valid=diag['valid'],acceleration=a.tolist(),shrink=shrink.tolist(),raw_estimate=diag))
