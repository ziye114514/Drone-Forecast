"""One joint p/v/net-acceleration state, matching the existing Singer rollout.

References: Stone Soup Singer model; FilterPy/Van Loan discretization.
Independent small adapter, no second/parallel trajectory filter. The inherited
3000px/s^2 acceleration cap supplies a fixed three-sigma prior (sigma=1000).
No trajectory labels or scene-specific parameter selection are used.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm

TAU=.35
SIGMA_ACCELERATION=3000./3.


@lru_cache(maxsize=64)
def transition(dt_us,tau=TAU):
    dt=dt_us*1e-6
    if not np.isfinite(tau) or tau<=0:raise ValueError('Positive acceleration correlation time required')
    A=np.array([[0.,1.,0.],[0.,0.,1.],[0.,0.,-1/tau]])
    G=np.zeros((3,3));G[2,2]=2*SIGMA_ACCELERATION**2/tau
    block=np.zeros((6,6));block[:3,:3]=-A;block[:3,3:]=G;block[3:,3:]=A.T
    exponential=expm(block*dt)
    F=exponential[3:,3:].T;Q=F@exponential[:3,3:]
    return np.kron(F,np.eye(2)),np.kron((Q+Q.T)*.5,np.eye(2))


class SingerPositionState:
    def __init__(self):
        self.kf=self;self.x=None;self.P=None;self.t=None;self.seen=None;self.observations=0;self.noise_scale=1.
        # Compile/load the fixed30Hz matrix-exponential/LAPACK path before the
        # source-time clock starts, rather than during a live measurement.
        transition(33333)
        np.linalg.solve(np.eye(2),np.ones((2,6)))

    def step(self,t,body):
        detected=bool(body.get('detected'));z=np.asarray(body['center'],float) if detected else None
        if self.seen is not None and t-self.seen>.5:self.__init__()
        innovation=None
        if self.x is None:
            if z is None:return None,False,dict(position_velocity_initial_state_only=False,joint_net_acceleration_state=True,reason='uninitialized')
            self.x=np.r_[z,[0.,0.,0.,0.]]
            self.P=np.diag([10000.,10000.,10000.,10000.,SIGMA_ACCELERATION**2,SIGMA_ACCELERATION**2])
        else:
            dt_us=round((t-self.t)*1e6)
            if dt_us<=0:raise ValueError('Strictly increasing native source cutoffs required')
            F,Q=transition(dt_us);self.x=F@self.x;self.P=F@self.P@F.T+Q
            if z is not None:
                residual=z-self.x[:2];S=self.P[:2,:2]+np.eye(2)*36.*self.noise_scale**2
                K=np.linalg.solve(S,self.P[:2,:]).T
                innovation=float(residual@np.linalg.solve(S,residual))
                self.x+=K@residual
                H=np.zeros((2,6));H[:,:2]=np.eye(2);I=np.eye(6)-K@H
                R=np.eye(2)*36.*self.noise_scale**2
                self.P=I@self.P@I.T+K@R@K.T
                self.P=(self.P+self.P.T)*.5
        self.t=t
        if detected:self.seen=t;self.observations+=1
        acceleration=self.x[4:].copy();norm=np.linalg.norm(acceleration)
        if norm>3000:acceleration*=3000/norm
        c=dict(position=self.x[:2].tolist(),velocity=self.x[2:4].tolist(),acceleration=acceleration.tolist())
        valid=self.observations>=5 and self.seen is not None and t-self.seen<=.16
        return c,bool(valid),dict(position_velocity_initial_state_only=False,joint_net_acceleration_state=True,
            parallel_kinematic_acceleration=False,acceleration_is_not_measured_rotor_thrust=True,
            correlation_s=TAU,prior_stationary_acceleration_sigma=SIGMA_ACCELERATION,
            measurement_innovation_chi2=innovation,observations=self.observations)
