"""Correlated force-rate extension of the original random-walk unknown force.

The old acceleration diffusion is2*sigma_a^2/tau. Set stationary jerk sigma to
sigma_a/tau, so integrated OU jerk has the SAME long-run acceleration diffusion.
No new fitted magnitude/time parameter. Acceleration need not revert to zero
when its current derivative vanishes. Missing-position frames keep173's short
Singer prior; no unobserved jerk is granted measurement authority.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm
from smooth_layout_force import SmoothLayoutForceObserver
from singer_position_state import transition as singer_transition
from rotor_force_observer import RotorForceObserver

TAU=.35
SIGMA_JERK=1000./TAU


@lru_cache(maxsize=64)
def transition(dt_us):
    dt=dt_us*1e-6
    A=np.array([[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.],[0.,0.,0.,-1/TAU]])
    G=np.zeros((4,4));G[3,3]=2*SIGMA_JERK**2/TAU
    M=np.zeros((8,8));M[:4,:4]=-A;M[:4,4:]=G;M[4:,4:]=A.T
    E=expm(M*dt);F=E[4:,4:].T;Q=F@E[:4,4:]
    return np.kron(F,np.eye(2)),np.kron((Q+Q.T)*.5,np.eye(2))


def coefficients(h):
    h=float(h);x=h/TAU
    j=h**3*(1/6-x/24+x*x/120-x**3/720) if abs(x)<.01 else TAU**3*(.5*x*x-x-np.expm1(-x))
    return .5*h*h,j


class ForceRateLayoutObserver(SmoothLayoutForceObserver):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.rate_unobserved=False;transition(33333)

    def _initialize(self,z):
        super()._initialize(z);self.P[11:13,11:13]=np.eye(2)*SIGMA_JERK**2

    def _transition(self,dt):
        F=np.eye(self.state_dim);Q=np.zeros((self.state_dim,self.state_dim))
        if self.rate_unobserved:
            A,Qnet=singer_transition(round(dt*1e6),.1);indices=np.arange(6)
            decay=np.exp(-dt/.1);F[11:13,11:13]*=decay
            Q[11:13,11:13]=np.eye(2)*SIGMA_JERK**2*(1-decay**2)
            offset=np.r_[np.zeros(4),self.u]
        else:
            A,Qnet=transition(round(dt*1e6));indices=self.dynamic_indices
            offset=np.r_[np.zeros(4),self.u,np.zeros(2)]
        F[np.ix_(indices,indices)]=A;Q[np.ix_(indices,indices)]=Qnet
        F[indices,self.gain_index]=A[:,4:6]@self.u-offset
        return F,Q

    def _joint_step(self,t,body,u,*args,**kwargs):
        self.rate_unobserved=not body.get('detected') or body.get('center_provisional',False)
        before=self.two_point_initialized;result=super()._joint_step(t,body,u,*args,**kwargs)
        if self.two_point_initialized and not before:self.P[11:13,11:13]=np.eye(2)*SIGMA_JERK**2
        result[-1].update(force_prior='integrated_OU_rate_matched_old_acceleration_diffusion',
            force_derivative_used_in_rollout=not self.rate_unobserved,
            missing_position_uses_short_Singer=True)
        return result

    def _force(self,active,acceleration):
        result=RotorForceObserver._force(self,active,acceleration)
        if self.rate_unobserved:
            result.update(acceleration_correlation_s=.1,future_net_acceleration_prior='Singer during missing position',
                force_derivative_unobserved=True)
        else:
            result.update(future_net_force_model='integrated_ou_rate',
                jerk_px_s3=[0.,0.] if self.x is None else self.x[11:13].tolist(),
                jerk_is_lumped_dynamics_not_measured_rotor_force_rate=True,
                future_net_acceleration_prior='unknown_force_level_with_correlated_derivative',
                force_rate_correlation_s=TAU)
        return result
