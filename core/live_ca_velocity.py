"""Isolated CA-velocity candidate, reusing the existing KinematicFilter.

Current body position, robust acceleration and output qualification remain
those of LiveMotion. The six-state filter contributes only current velocity.
No force feature, manual label or future event enters either state update.
"""
from types import SimpleNamespace
import numpy as np
from live_motion import LiveMotion
from event_drone_forecast.motion import KinematicFilter
from event_drone_forecast.position.kalman import PARAMS


class CAVelocityMotion(LiveMotion):
    def __init__(self,tracker_hz,noise_model='legacy'):
        super().__init__(tracker_hz,noise_model)
        # Existing order3 uses jerk sigma=cfg.acceleration_sigma*5.
        # Match the original discrete-Q velocity variance at the requested
        # nominal interval: (sigma_j*dt^2/2)^2=(sigma_a*dt)^2.
        self.jerk_sigma=2*PARAMS['sigma_acceleration']/self.period
        self.ca=KinematicFilter(SimpleNamespace(position_sigma=PARAMS['sigma_body_px'],
            acceleration_sigma=self.jerk_sigma/5.),order=3)
        self.ca_owner=self.kf

    def step(self,t,body):
        c,valid,d=super().step(t,body)
        # Parent identity/gap reset reinitializes this subclass as well.
        assert self.ca_owner is self.kf
        initial=self.ca.x is None
        z=body['center'] if body.get('detected') else None
        state=self.ca.update(t,z,confidence=1/d['measurement_noise_scale'])
        if initial and state is not None:
            # Match original p/v priors, give unobserved acceleration the
            # existing sigma_acceleration prior. No covariance fitting.
            self.ca.P=np.diag([PARAMS['initial_covariance']]*4+
                [PARAMS['sigma_acceleration']**2]*2)
        used=c is not None and state is not None
        if used:c['velocity']=state[2:4].tolist()
        d.update(ca_velocity_used=used,jerk_sigma_px_s3=self.jerk_sigma,
            ca_state_acceleration=state[4:6].tolist() if state is not None else None)
        return c,valid,d
