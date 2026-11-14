"""Existing4-state position KF driven by the previous physical acceleration.

Deterministic control changes the mean, not the existing process/measurement
noise. No second motion predictor or new tuning constants are introduced.
"""
import numpy as np
from live_structure_bridge import NoiseKF


class ForceDrivenNoiseKF(NoiseKF):
    def __init__(self,noise_model='legacy'):
        super().__init__(noise_model);self.control=np.zeros(2);self.used_control=np.zeros(2)

    def set_control(self,acceleration):
        a=np.asarray(acceleration,float)
        if a.shape!=(2,) or not np.isfinite(a).all():raise ValueError('Finite2D acceleration required')
        length=np.linalg.norm(a);self.control=a*min(1.,3000./max(length,1e-9))

    def step(self,t,z,measurement_scale=1.):
        self.used_control=np.zeros(2)
        if self.x is not None and self.t is not None and 0.<t-self.t<=.12:
            dt=t-self.t;self.used_control=self.control.copy()
            # F^{-1}B a on the prior mean, followed by the UNCHANGED parent
            # prediction/update, is exactly F x+B a. B=[dt²/2 I;dt I].
            self.x[:2]-=.5*dt*dt*self.control;self.x[2:4]+=dt*self.control
        return super().step(t,z,measurement_scale)
