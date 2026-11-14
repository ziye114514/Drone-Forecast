"""Position-observed extended state: [p, v, total disturbance acceleration].

Known/assumed acceleration is an input; gravity, drag and unmodeled inputs can
be lumped into d. Pixel-domain d is not a force measurement or a wind estimate.
Continuous white-jerk covariance; uses the existing pinned FilterPy wrappers.
"""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.position._filterpy import predict,update

PARAMS=dict(reset_gap_s=.5,position_sigma_px=6.,acceleration_cap=3000.,
    jerk_sigma=6400./np.sqrt(.35),minimum_observations=6,minimum_span_s=.16,
    innovation_chi2_99=9.21034037197618,
    uncertainty_is_proxy=True,force_in_newtons=False,wind_or_mass_identified=False)


class LumpedDisturbance:
    def __init__(self,measurement_sigma=6.,jerk_sigma=None):
        self.sigma=float(measurement_sigma);self.jerk_sigma=PARAMS['jerk_sigma'] if jerk_sigma is None else float(jerk_sigma)
        self.x=None;self.P=None;self.last_t=None;self.seen=None;self.started=None
        self.observations=0;self.previous_u=np.zeros(2);self.input_valid=False;self.identity=None

    def step(self,t,position,known_acceleration=None,identity=None):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Strictly increasing cutoffs required')
        z=None if position is None else np.asarray(position,float)
        u=np.zeros(2) if known_acceleration is None else np.asarray(known_acceleration,float)
        if u.shape!=(2,) or not np.isfinite(u).all():raise ValueError('Finite2D input required')
        if z is not None and (z.shape!=(2,) or not np.isfinite(z).all()):raise ValueError('Finite2D position required')
        valid=known_acceleration is not None
        reset=(identity is not None and self.identity is not None and identity!=self.identity)
        reset=reset or (self.seen is not None and t-self.seen>.5)
        if reset:
            self.x=None;self.P=None;self.observations=0;self.started=None
        if identity is not None:self.identity=identity
        accepted=False;nis=None
        if self.x is None:
            if z is not None:
                self.x=np.r_[z,np.zeros(4)];self.P=np.diag([self.sigma**2]*2+[1e4]*2+[1e6]*2)
                self.started=t;self.seen=t;self.observations=1;accepted=True
        else:
            dt=t-self.last_t;I=np.eye(2);F=np.eye(6)
            F[:2,2:4]=dt*I;F[:2,4:6]=.5*dt*dt*I;F[2:4,4:6]=dt*I
            B=np.r_[.5*dt*dt*I,dt*I,np.zeros((2,2))]
            Q1=np.array([[dt**5/20,dt**4/8,dt**3/6],
                         [dt**4/8,dt**3/3,dt**2/2],[dt**3/6,dt**2/2,dt]])
            self.x,self.P=predict(self.x,self.P,F=F,Q=self.jerk_sigma**2*np.kron(Q1,I))
            self.x+=B@self.previous_u
            H=np.c_[I,np.zeros((2,4))];R=I*self.sigma**2
            if z is not None:
                innovation=z-H@self.x;S=H@self.P@H.T+R
                nis=float(innovation@np.linalg.solve(S,innovation))
                if nis<=PARAMS['innovation_chi2_99']:
                    self.x,self.P=update(self.x,self.P,z,R=R,H=H)
                    self.seen=t;self.observations+=1;accepted=True
            if valid!=self.input_valid:
                # Availability changes are changes of decomposition, not a
                # sudden observed acceleration. Keep total acceleration fixed.
                self.x[4:6]-=u-self.previous_u
            self.x[4:6]=np.clip(self.x[4:6],-3000.,3000.)
        self.last_t=t;self.previous_u=u.copy();self.input_valid=valid
        ready=bool(self.x is not None and self.observations>=6 and t-self.started>=.16
                   and self.seen is not None and t-self.seen<=.16)
        coefficient=None if self.x is None else dict(position=self.x[:2].tolist(),velocity=self.x[2:4].tolist(),
            acceleration=np.clip(u+self.x[4:6],-3000.,3000.).tolist())
        diagnostic=dict(ready=ready,accepted_position=accepted,innovation_nis=nis,
            disturbance_acceleration=None if self.x is None else self.x[4:6].tolist(),
            input_acceleration=u.tolist(),input_available=valid,
            disturbance_includes_unobserved_thrust=not valid,observations=self.observations,
            force_calibrated=False,uncertainty_is_proxy=True)
        return coefficient,diagnostic


def forecasts(coefficient):
    if coefficient is None:return {}
    p,v,a=(np.asarray(coefficient[k]) for k in ['position','velocity','acceleration'])
    return {str(h):(p+h*v+.5*h*h*a).tolist() for h in (.1,.2,.4)}


class DelayedResidualDisturbance:
    """Reuse149 delayed position updates, keep trusted baseline p/v intact.

    This estimates residual image acceleration beyond the kinematic rollout,
    including gravity/drag/thrust-gain mismatch left by its nominal model. It
    is deliberately distinct from a Newton-force or separately identified wind.
    """
    def __init__(self):
        sys.path[:0]=[str(ROOT/'research/stage149'),str(ROOT/'research/stage148')]
        from dynamics149 import Core
        self.constructor=Core;self.core=Core();self.identity=None;self.last_t=None

    def step(self,t,body,baseline,assumed_acceleration=None,identity=None):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Strictly increasing cutoffs required')
        self.last_t=t
        if identity is not None and self.identity is not None and identity!=self.identity:self.core=self.constructor()
        if identity is not None:self.identity=identity
        self.core.mature(t,body)
        u=np.zeros(2) if assumed_acceleration is None else np.asarray(assumed_acceleration,float)
        proposals={};fused={};weights={};pairs={}
        for h in (.1,.2,.4):
            key=str(h)
            if key not in baseline:continue
            base=np.asarray(baseline[key]);A=.5*h*h*np.array([[0.,1.,0.],[0.,0.,1.]])
            known=base+.5*h*h*u
            # The first149 parameter is unused/frozen: no simultaneous free
            # identification of gain and arbitrary disturbance from one stream.
            candidate=known+A@self.core.theta;delta=candidate-base
            candidate=base+delta*min(1.,40./max(np.linalg.norm(delta),1e-9))
            loss=list(self.core.loss[h]);weight=0.
            if self.core.updates>=10 and len(loss)>=10:
                means=np.mean(np.asarray([q[1:] for q in loss]),axis=0)
                weight=min(.35,max(0.,(means[0]-means[2])/max(means[0],1.)))
            proposals[key]=candidate.tolist();fused[key]=((1-weight)*base+weight*candidate).tolist()
            weights[key]=float(weight);pairs[key]=len(loss)
            if body.get('detected'):
                self.core.queue.append(dict(target_t=t+h,h=h,A=A,known=known,
                    baseline=base.copy(),bias=base.copy(),physical=candidate.copy()))
        from collections import deque
        self.core.queue=deque(sorted(self.core.queue,key=lambda q:q['target_t']))
        return dict(predictions=fused,proposals=proposals,weights=weights,past_pairs=pairs,
            residual_acceleration=self.core.theta[1:].tolist(),updates=self.core.updates,
            current_position_velocity_preserved=True,uses_only_matured_body_observations=True,
            gain_identified=False,force_calibrated=False,
            nominal_model='existing kinematic prediction; correction is residual image acceleration')
