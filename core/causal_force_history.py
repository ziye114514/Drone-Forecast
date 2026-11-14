"""Past-only net-force history, explicitly separate from measured blade thrust.

The .35s window and .16s minimum span are inherited from plane trend, not
selected on these recordings. No second position estimator is introduced.
This is a correlated acceleration-history feature, not an independent sensor.
"""
import copy
from collections import deque
import numpy as np


class CausalForceHistory:
    def __init__(self):self.rows=deque();self.last=None
    def step(self,row):
        t=row['t'];r=copy.deepcopy(row)
        if self.last is not None and t-self.last>.12:self.rows.clear()
        self.last=t
        c=row['coefficient']
        if c is not None and row['valid'] and row['motion'].get('current_observation'):
            self.rows.append((t,np.asarray(c['acceleration'])))
        while self.rows and t-self.rows[0][0]>.35+1e-8:self.rows.popleft()
        info=dict(qualified=False,source='past_correlated_net_force_states',actual_rotor_thrust_rate_measured=False)
        if len(self.rows)>=6 and self.rows[-1][0]-self.rows[0][0]>=.16-1e-8 and abs(t-self.rows[-1][0])<1e-6:
            times=np.array([q[0]-t for q in self.rows]);a=np.stack([q[1] for q in self.rows]);X=np.c_[np.ones(len(times)),times]
            b=np.linalg.lstsq(X,a,rcond=None)[0];residual=a-X@b
            info.update(qualified=True,net_jerk_px_s3=b[1].tolist(),span_s=float(np.ptp(times)),samples=len(times),
                residual_rms_px_s2=np.sqrt(np.mean(residual**2,axis=0)).tolist(),uncertainty_not_independent_IMU=True)
        r['causal_force_history']=info;return r
