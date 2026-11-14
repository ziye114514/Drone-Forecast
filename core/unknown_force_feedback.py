"""When event thrust is unavailable, estimate TOTAL unknown image force.

Only existing current positions supervise a bounded local acceleration fit.
No hidden rotor rate, independent force truth or second trajectory KF is added.
The same estimate seeds reacquisition instead of forcing acceleration to zero.
"""
from collections import deque
import numpy as np
from responsive_force import DampedResponsiveForce


class UnknownForceFeedback(DampedResponsiveForce):
    def __init__(self,default_gain=200.,early_disturbance=True,hold_acceleration_gaps=True,gain_signal_to_noise=0.):
        super().__init__(default_gain,early_disturbance,hold_acceleration_gaps,gain_signal_to_noise)
        self.position_window=deque()

    def step(self,t,c,observed_position,u,quality=0.):
        if self.last_t is not None and t<self.last_t:raise ValueError('Causal timestamps required')
        if observed_position is not None:self.position_window.append((t,np.asarray(observed_position,float)))
        while self.position_window and t-self.position_window[0][0]>.35:self.position_window.popleft()
        estimate=None;audit=dict(qualified=False,source='current_position_total_unknown_acceleration',
            estimates_unobserved_thrust_when_period_missing=True,metric_force_measured=False)
        if len(self.position_window)>=7 and t-self.position_window[-1][0]<=.12:
            tt=np.array([r[0] for r in self.position_window]);zz=np.array([r[1] for r in self.position_window]);span=tt[-1]-tt[0]
            if span>=.2-1e-5 and np.max(np.diff(tt))<=.15:
                scaled=(tt-tt[-1])/span;A=np.c_[np.ones(len(tt)),scaled,.5*scaled**2]
                beta=np.linalg.lstsq(A,zz,rcond=None)[0];residual=np.linalg.norm(A@beta-zz,axis=1)
                weights=np.minimum(1.,6./np.maximum(residual,1e-9));Aw=A*np.sqrt(weights)[:,None]
                beta=np.linalg.lstsq(Aw,zz*np.sqrt(weights)[:,None],rcond=None)[0]
                errors=A@beta-zz;rms=float(np.sqrt(np.mean(errors**2)));condition=float(np.linalg.cond(Aw))
                if rms<=12. and condition<=1500.:
                    raw=beta[2]/span**2
                    sigma2=np.maximum(36.,np.sum(errors**2,axis=0)/max(1,len(tt)-3))
                    covariance_factor=float(np.linalg.pinv(Aw.T@Aw,rcond=1e-10)[2,2])/span**4
                    variance=sigma2*covariance_factor;energy=float(raw@raw)
                    shrink=max(0.,1.-float(variance.sum())/max(energy,1e-9))
                    estimate=raw*shrink
                    length=np.linalg.norm(estimate)
                    if length>3000.:estimate*=3000./length
                    audit.update(qualified=True,observations=len(tt),span_s=float(span),rms_px=rms,
                        raw_total_acceleration_px_s2=raw.tolist(),acceleration_variance_proxy=variance.tolist(),
                        shrinkage=float(shrink),estimate_px_s2=estimate.tolist(),noise_proxy_not_calibrated=True,
                        trusted_position_velocity_unchanged=True)
        good=bool(c is not None and u is not None and np.isfinite(u).all() and quality>=.35)
        reacquiring=bool(good and (self.last_valid is None or t-self.last_valid>.12))
        fallback=bool(estimate is not None and c is not None and (not good or reacquiring))
        supplied=c
        if fallback:supplied=dict(c,acceleration=estimate.tolist())
        predictions,info=super().step(t,supplied,observed_position,u,quality)
        info.update(total_unknown_force=audit,total_unknown_force_seed_supplied=fallback,
            reacquisition_seed_from_observed_motion=bool(fallback and reacquiring),
            event_thrust_still_unobserved=not good)
        if fallback and not info['active'] and not info['short_gap_acceleration_held']:
            info['acceleration_source']='total_unknown_force_from_observed_position_residual'
            info['disturbance_px_s2']=estimate.tolist()
        return predictions,info
