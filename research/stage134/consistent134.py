"""Optimize acceleration rollout using current motion consistency.

The old23 shrink was applied before the acceleration cap. If the estimate is
clipped below its own uncertainty, it can still generate an unjustified large
forecast. Here cap first, then shrink. A long-window acceleration is used only
to the magnitude/direction supported by the latest80ms velocity trend.
"""
from pathlib import Path
from collections import deque
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage132')]
from fast132 import Fast,forecast,cap
PARAMS=dict(cap_before_uncertainty_shrink=True,velocity_trend_window_s=.08,minimum_trend_span_s=.04,minimum_trend_points=3,maximum_gap_s=.04000001,
    rule='Cap robust acceleration first, shrink by its standard-error variance. Keep each component only if current short-window velocity slope agrees; bound its magnitude by that slope. No label-derived threshold.',
    positions='Same fast107 position state; no first-frame rotor position offset. Geometry updates relative velocity only.',
    scope='Optimize forecast model, no geometric-gate relaxation. Current and past observations only. Standard confidence/consistency reasoning, not a novelty assertion.')
class Predictor:
    def __init__(self):self.fast=Fast();self.vel={n:deque() for n in ['body','geometry']};self.last_body=None
    def step(self,t,body,models):
        if self.last_body is not None and t-self.last_body>.5:self.__init__()
        if body.get('detected'):
            if self.last_body is not None and t-self.last_body>PARAMS['maximum_gap_s']:
                self.fast.history.clear()
                for h in self.vel.values():h.clear()
            self.last_body=t
        f=self.fast.step(t,body,models);fit=f['acceleration_diagnostic']['raw_estimate']
        if fit.get('valid'):
            capped=cap(fit['acceleration']);se=np.asarray(fit['standard_error']);shrink=np.maximum(0.,1.-se*se/(capped*capped+1e-12));trusted=capped*shrink
        else:trusted=np.zeros(2);shrink=np.zeros(2)
        out={}
        for name,source in [('body','fast_body_acc'),('geometry','fast_geometry_acc')]:
            c=copy.deepcopy(f[source]['coefficient']);h=self.vel[name];short=np.zeros(2);supported=False
            if c is not None and body.get('detected'):h.append((t,np.array(c['velocity'])))
            while h and t-h[0][0]>PARAMS['velocity_trend_window_s']+1e-8:h.popleft()
            if len(h)>=3 and t-h[0][0]>=PARAMS['minimum_trend_span_s']-1e-8:
                ts=np.array([s-t for s,_ in h]);vv=np.array([v for _,v in h]);u=ts-ts.mean();short=np.sum(u[:,None]*(vv-vv.mean(0)),axis=0)/np.sum(u*u);supported=True
            a=np.where(trusted*short>0,np.sign(trusted)*np.minimum(abs(trusted),abs(short)),0.) if supported else np.zeros(2)
            if c is not None:c['acceleration']=a.tolist()
            out['consistent_'+name]=dict(coefficient=c,source=f[source]['source'],short_acceleration=short.tolist(),used_acceleration=a.tolist(),uncertainty_shrink=shrink.tolist())
        return out
