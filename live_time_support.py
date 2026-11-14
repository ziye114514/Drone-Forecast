"""Experimental actual-time support for irregularly delivered measurements.

The KF, robust fit, thresholds and rollout are reused unchanged. This separates
missed compute slots from explicit no-detection observations; no default change.
"""
import copy
import numpy as np
from live_motion import LiveMotion
from event_drone_forecast.position.robust import cap


class TimeSupportMotion(LiveMotion):
    def step(self,t,body):
        coefficient,valid,diagnostic=super().step(t,body)
        support=list(self.support)
        stamps=np.asarray([s[0] for s in support],float)
        observed=np.asarray([s[1] for s in support],float)
        coverage=0.
        if len(stamps):
            left=max(t-.35,stamps[0]-.5*self.period)
            x=np.r_[left,stamps];y=np.r_[observed[0],observed]
            coverage=float(np.clip(np.trapezoid(y,x=x)/.35,0.,1.))
        fit=diagnostic['fit']
        eligible=bool(body.get('detected') and diagnostic['strong_observation'] and coverage>=.9
            and diagnostic['span_s']>=.35-self.period-1e-8
            and diagnostic['maximum_gap_s']<=2*self.period+1e-8
            and all(ok for _,seen,ok in support if seen)
            and fit.get('valid') and fit['median_abs_residual_px']<=6.
            and fit['robust_inlier_fraction']>=.9)
        if coefficient is not None:
            coefficient=copy.deepcopy(coefficient)
            if eligible:
                a=np.asarray(fit['acceleration']);se=np.asarray(fit['standard_error'])
                shrink=np.maximum(0.,1.-se*se/(a*a+1e-12))
                coefficient['acceleration']=cap(a*shrink).tolist()
            else:coefficient['acceleration']=[0.,0.]
        diagnostic=dict(diagnostic,acceleration_eligible_count_policy=diagnostic['acceleration_eligible'],
            acceleration_eligible=eligible,time_support_coverage=coverage,
            support_policy='trapezoidal integration over actual delivered timestamps')
        return coefficient,valid,diagnostic
