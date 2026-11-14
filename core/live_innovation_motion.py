"""Experimental, past-innovation-only Q scale; not a deployed default.

Reuses the pinned FilterPy predict/update and current live qualification/rollout.
Only process covariance intensity changes, with the same discrete Q shape.
This is a small generic innovation-adaptation probe, not a reproduction of
arXiv:2112.12082 and not an event-camera-specific novelty claim.
"""
import math
import numpy as np
from live_motion import LiveMotion
from event_drone_forecast.position.kalman import MeasurementKF,PARAMS,predict,update,coefficient

DESIGN=dict(initial_Q_scale=1.,minimum_Q_scale=.01,maximum_Q_scale=1.,
    innovation_ratio_bounds=[.25,4.],expected_NIS=2.,
    maneuver_threshold_NIS=-2*math.log(.01),smoothing_tau_s=PARAMS['smooth_tau_s'],
    current_innovation_changes_next_tick_Q_only=True,adaptive_R=False,
    parameters_swept=False,default_promoted=False)


class InnovationKF(MeasurementKF):
    def __init__(self,noise_model='legacy',enabled=True):
        if noise_model!='legacy':raise ValueError('This isolated probe uses legacy Q only')
        self.enabled=enabled;self.Q_scale=1.;self.allow_adaptation=False
        self.last_nis=None;self.Q_scale_used=1.;self.maneuver_resets=0
        super().__init__(noise_model)

    def step(self,t,z,measurement_scale=1.):
        if self.seen is not None and t-self.seen>PARAMS['reset_body_gap_s']:
            self.__init__(self.noise_model,self.enabled)
        if self.x is None:
            if z is None:return None
            self.x=np.r_[z,[0.,0.]];self.P=np.eye(4)*PARAMS['initial_covariance']
            self.t=t;self.seen=t;return coefficient(self.x)
        dt=t-self.t
        if dt<=0:raise ValueError('Strictly increasing timestamps required')
        self.t=t;F=np.eye(4);F[:2,2:4]=np.eye(2)*dt
        B=np.r_[np.eye(2)*(.5*dt*dt),np.eye(2)*dt]
        self.Q_scale_used=self.Q_scale if self.enabled else 1.
        Q=PARAMS['sigma_acceleration']**2*(B@B.T)*self.Q_scale_used
        self.x,self.P=predict(self.x,self.P,F=F,Q=Q)
        self.last_nis=None
        if z is not None:
            self.seen=t;H=np.c_[np.eye(2),np.zeros((2,2))]
            R=np.eye(2)*36.*measurement_scale**2
            residual=np.asarray(z)-H@self.x;S=H@self.P@H.T+R
            self.last_nis=float(residual@np.linalg.solve(S,residual))
            self.x,self.P=update(self.x,self.P,z,R=R,H=H)
            if self.enabled and self.allow_adaptation:
                if self.last_nis>DESIGN['maneuver_threshold_NIS']:
                    self.Q_scale=1.;self.maneuver_resets+=1
                else:
                    ratio=float(np.clip(self.last_nis/2.,.25,4.))
                    beta=1-math.exp(-dt/DESIGN['smoothing_tau_s'])
                    self.Q_scale=float(np.clip(self.Q_scale*ratio**beta,.01,1.))
        return coefficient(self.x)


class InnovationMotion(LiveMotion):
    def __init__(self,tracker_hz,noise_model='legacy',enabled=None):
        # LiveMotion calls __init__ again on an identity or long-gap reset.
        # Preserve explicit disabled controls across those resets.
        self.enabled=getattr(self,'enabled',True) if enabled is None else bool(enabled)
        super().__init__(tracker_hz,noise_model)
        self.kf=InnovationKF(noise_model,self.enabled)

    def step(self,t,body):
        b=np.asarray(body.get('bbox',[]),float)
        complete=b.shape==(4,) and np.isfinite(b).all() and np.all(b[:2]>0) and np.all(b[2:]<[1279,719])
        self.kf.allow_adaptation=bool(body.get('detected') and not body.get('low_confidence',False)
            and complete and self.observations>=5)
        c,valid,d=super().step(t,body)
        d.update(innovation_NIS=self.kf.last_nis,Q_scale_used=self.kf.Q_scale_used,
            Q_scale_next=self.kf.Q_scale,maneuver_Q_resets=self.kf.maneuver_resets,
            innovation_adaptation_enabled=self.enabled)
        return c,valid,d
