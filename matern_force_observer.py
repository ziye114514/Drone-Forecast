"""Shared two-force state with a correlated net-force derivative.

Matérn3/2 state-space prior (Hartikainen/Särkkä2010), independently implemented.
Same stationary acceleration variance and integrated correlation as Singer.
The derivative belongs to lumped unknown dynamics, not a measured rotor torque.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm
from rotor_force_observer import RotorForceObserver

RATE=2/.35


@lru_cache(maxsize=64)
def transition(dt_us):
    dt=dt_us*1e-6;lam=RATE
    A=np.array([[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.],[0.,0.,-lam*lam,-2*lam]])
    G=np.zeros((4,4));G[3,3]=4*lam**3*1000.**2
    M=np.zeros((8,8));M[:4,:4]=-A;M[:4,4:]=G;M[4:,4:]=A.T
    E=expm(M*dt);F=E[4:,4:].T;Q=F@E[:4,4:]
    return np.kron(F,np.eye(2)),np.kron((Q+Q.T)*.5,np.eye(2))


def coefficients(h):
    x=RATE*float(h);e=np.expm1(-x)
    if abs(x)<1e-3:
        # Native sub-tick publication ages need cancellation-safe integrals.
        a=.5*h*h-RATE**2*h**4/24+RATE**3*h**5/60
        j=h**3/6-RATE*h**4/12+RATE**2*h**5/40
    else:
        a=(3*e+x*(e+3))/RATE**2
        j=(x*(e+2)+2*e)/RATE**3
    return a,j


class MaternForceObserver(RotorForceObserver):
    state_dim=9
    gain_index=8

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.startup_fit or self.two_point_init:raise ValueError('Jerk-state startup fit is not implemented')
        self.consistent_dynamics=True;transition(33333)

    def _initialize(self,z):
        self.x=np.r_[z,np.zeros(6),self.default_gain]
        self.P=np.diag([10000.]*2+[self.velocity_sigma**2]*2+[1e6]*2+[RATE**2*1e6]*2+[self.default_gain**2])

    def _transition(self,dt):
        A,Qnet=transition(round(dt*1e6));F=np.eye(9);Q=np.zeros((9,9))
        F[:8,:8]=A;Q[:8,:8]=Qnet
        F[:8,8]=A[:,4:6]@self.u-np.r_[np.zeros(4),self.u,np.zeros(2)]
        return F,Q

    def _force(self,active,acceleration):
        result=super()._force(active,acceleration)
        result.update(future_net_force_model='matern32',force_decay_rate=RATE,
            jerk_px_s3=[0.,0.] if self.x is None else self.x[6:8].tolist(),
            jerk_is_lumped_dynamics_not_measured_rotor_force_rate=True,
            future_net_acceleration_prior='matched_Matern32_net_force_and_derivative',
            integrated_acceleration_correlation_s=.35)
        return result
