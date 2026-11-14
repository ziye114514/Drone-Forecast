"""Conditional perspective rollout from matched current propeller major axes.

An orthographic projected circle's major axis is insensitive to plane tilt;
partial/uncertain conics are excluded. Scale is relative, not metric depth.
"""
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment
from adaptive_force_memory import kernel


class RotorScaleRate:
    def __init__(self):self.previous=None;self.history=deque();self.level=0.

    def step(self,t,geometry,motion):
        current=[r for r in geometry.get('rotors',[]) if r.get('current_observation',True)
            and not r.get('fov_clipped') and not r.get('association_ambiguous')
            and r.get('mode')=='ellipse' and r.get('geometry_quality',0)>=.6 and r.get('ratio',0)>.16]
        shift=(motion or {}).get('estimate');deltas=[]
        if self.previous is not None and shift is not None and len(current)>=2:
            old_t,old=self.previous;dt=t-old_t
            if 0<dt<=.12:
                delta=np.array([shift['dx'],shift['dy']])
                cost=np.linalg.norm(np.array([[r['cx'],r['cy']] for r in old])[:,None]+delta-
                    np.array([[r['cx'],r['cy']] for r in current])[None],axis=2)
                ii,jj=linear_sum_assignment(cost)
                for i,j in zip(ii,jj):
                    if cost[i,j]<=max(6.,.15*old[i]['major']):
                        deltas.append(float(np.log(current[j]['major']/old[i]['major'])))
        self.previous=(t,current) if len(current)>=2 else None
        if len(deltas)<2 or np.ptp(deltas)>.08:
            self.history.clear();self.level=0.
            return dict(applied=False,reason='no_consistent_current_matched_circle_sizes')
        self.level+=float(np.median(deltas));self.history.append((t,self.level))
        while self.history and t-self.history[0][0]>.35:self.history.popleft()
        if len(self.history)<6 or t-self.history[0][0]<.16:
            return dict(applied=False,reason='scale_history_warming',pairs=len(deltas))
        x=np.array([p[0] for p in self.history]);y=np.array([p[1] for p in self.history])
        i,j=np.triu_indices(len(x),1);slopes=(y[j]-y[i])/(x[j]-x[i])
        rate=float(np.median(slopes));scatter=float(1.4826*np.median(abs(slopes-rate)))
        shrink=rate*rate/(rate*rate+scatter*scatter+1e-12)
        return dict(applied=True,rate_per_s=rate*shrink,raw_rate_per_s=rate,
            rate_scatter_proxy=scatter,confidence_is_calibrated=False,
            matched_current_pairs=len(deltas),metric_depth_measured=False,
            assumption='constant relative depth velocity and stable physical rotor radius')


def projective_forecast(row,h):
    info=row.get('rotor_projective_rollout') or {}
    if not info.get('applied'):return None
    rate=float(info['rate_per_s'])
    if abs(rate*h)>.5:return None
    c=row['coefficient'];p=np.asarray(c['position']);v=np.asarray(c['velocity'])
    a=np.asarray(row['force']['acceleration_px_s2']);tau=row['force']['acceleration_correlation_s']
    # p''_image contains2*(scale'/scale)*p'_image under constant depth velocity.
    # Remove that term before rolling the effective lateral force, then project.
    return p+(v*h+kernel(-1/tau,h)*(a-2*rate*v))/(1-rate*h)
