"""Damped-acceleration prediction with optional relative rotor velocity.

Exact Singer transition/process covariance via cached Van Loan exponential.
Image-plane dynamics, not identified mass/thrust or metric acceleration.
"""
from pathlib import Path
from functools import lru_cache
import sys
import numpy as np
from scipy.linalg import expm
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage130')]
from motion130 import RelativeVelocity,update,KF
PARAMS=dict(acceleration_decay_tau_s=.2,stationary_acceleration_sigma_px_s2=6400.,forecast_acceleration_cap_px_s2=2500.,
    source='tau=.2s existing physical-adaptation memory; sigma6400 original107 CV process scale; cap2500 original paper config. Fixed before131 outcomes; no endpoint grid search.',
    motion='Same130 geometrical-center velocity, invariant to constant source offset. Optional second update on one six-state filter, body position remains absolute reference.',
    formula='p(h)=p+h*v+(tau*h-tau^2*(1-exp(-h/tau)))*a; acceleration mean decays, not indefinite quadratic extrapolation.',
    limits='No true acceleration/RPM/COM truth; process scales heuristic. Exposed development clips, no independent test claim.')
@lru_cache(maxsize=16)
def transition(dt):
    tau=PARAMS['acceleration_decay_tau_s'];A=np.array([[0,1,0],[0,0,1],[0,0,-1/tau]],float);noise=np.zeros((3,3));noise[2,2]=2*PARAMS['stationary_acceleration_sigma_px_s2']**2/tau
    V=np.block([[A,noise],[np.zeros((3,3)),-A.T]])*dt;E=expm(V);F=E[:3,:3];Q=E[:3,3:]@F.T
    return np.kron(F,np.eye(2)),np.kron((Q+Q.T)/2,np.eye(2))
def forecast(c,h):
    if c is None:return None
    tau=c['decay_tau_s'];factor=tau*h-tau*tau*(-np.expm1(-h/tau))
    return (np.array(c['position'])+h*np.array(c['velocity'])+factor*np.array(c['acceleration'])).tolist()
class Damped:
    def __init__(self,use_geometry):self.use_geometry=use_geometry;self.x=None;self.P=None;self.t=None;self.seen=None;self.motion=RelativeVelocity()
    def step(self,t,body,models):
        if self.seen is not None and t-self.seen>.5:self.__init__(self.use_geometry)
        z=np.array(body['center']) if body.get('detected') else None
        if z is not None:self.seen=t
        slopes={k:self.motion.slope(k,t,models.get(k,{})) for k in ['count','occupancy']} if self.use_geometry else {}
        if self.x is None:
            if z is None:return dict(coefficient=None,source=None)
            self.x=np.r_[z,[0.,0.,0.,0.]];self.P=np.diag([10000.]*4+[PARAMS['stationary_acceleration_sigma_px_s2']**2]*2);self.t=t
        else:
            dt=round(t-self.t,8);assert dt>0;self.t=t;F,Q=transition(dt);self.x=F@self.x;self.P=F@self.P@F.T+Q
            if z is not None:self.x,self.P=update(self.x,self.P,z,R=np.eye(2)*36.,H=np.c_[np.eye(2),np.zeros((2,4))])
        source='count' if slopes.get('count') is not None else 'occupancy' if slopes.get('occupancy') is not None else None
        if source and z is not None:
            m=slopes[source];self.x,self.P=update(self.x,self.P,np.array(m['velocity']),R=np.eye(2)*m['variance'],H=np.c_[np.zeros((2,2)),np.eye(2),np.zeros((2,2))])
        a=self.x[4:].copy();norm=np.linalg.norm(a);cap=PARAMS['forecast_acceleration_cap_px_s2']
        if norm>cap:a*=cap/norm
        c=dict(position=self.x[:2].tolist(),velocity=self.x[2:4].tolist(),acceleration=a.tolist(),decay_tau_s=PARAMS['acceleration_decay_tau_s'])
        return dict(coefficient=c,source=source,state_acceleration=self.x[4:].tolist(),forecast_acceleration_capped=bool(norm>cap))
class Optimized131:
    def __init__(self):self.models={False:Damped(False),True:Damped(True)}
    def step(self,t,body,models):return {('damped_geometry' if k else 'damped_body'):m.step(t,body,models) for k,m in self.models.items()}
