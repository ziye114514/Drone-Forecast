"""Shared image errors in successive native relative-displacement factors.

delta_y=p_current-p_previous+e_current-e_previous+independent_noise.
Only the original4px^2 floor is split into two2px^2 endpoint priors.
Extra cycle/polarity uncertainty and the entire mechanical model stay intact.
This is an approximate observation-noise model, not a new force measurement.
"""
import numpy as np
from normal_response_observer import NormalResponseObserver
from separated_force_response import SeparatedForceObserver
from causal_response_credit import CausalResponseCredit


class EndpointNoiseMixin:
    endpoint_base=None
    endpoint_variance=2.

    def _initialize(self,z):
        super()._initialize(z)
        a=self.endpoint_base
        self.P[a:a+4,a:a+4]=np.eye(4)*self.endpoint_variance

    def _transition(self,dt):
        F,Q=super()._transition(dt);a=self.endpoint_base
        # Previous endpoint remains a conditional state; the new image has
        # an independent zero-mean endpoint prior, including after a gap.
        F[a+2:a+4]=0.;Q[a+2:a+4]=0.;Q[:,a+2:a+4]=0.
        Q[a+2:a+4,a+2:a+4]=np.eye(2)*self.endpoint_variance
        return F,Q

    def _clone_position(self):
        super()._clone_position();a=self.endpoint_base
        # The inherited two-point startup deliberately resets the covariance.
        # Reinstall only the endpoint prior if that reset cleared it.
        if not np.any(self.P[a+2:a+4,a+2:a+4]):
            self.P[a+2:a+4,a+2:a+4]=np.eye(2)*self.endpoint_variance
        J=np.eye(self.state_dim);J[a:a+2]=0.;J[a:a+2,a+2:a+4]=np.eye(2)
        self.x[a:a+2]=self.x[a+2:a+4];self.P=J@self.P@J.T

    def _measurement(self,z,body,motion,lag):
        H,R,value,flow=super()._measurement(z,body,motion,lag);a=self.endpoint_base
        if lag:raise ValueError('Shared endpoint model requires current native factors')
        H=H.copy();R=R.copy()
        if self.relative_used:
            H[:2,a:a+2]=-np.eye(2);H[:2,a+2:a+4]=np.eye(2)
            R[:2,:2]-=2*self.endpoint_variance*np.eye(2)
        else:
            H[:2,a+2:a+4]=np.eye(2)
            R[:2,:2]-=self.endpoint_variance*np.eye(2)
        # Subtract only the declared floor, not any measured extra variance.
        if np.linalg.eigvalsh((R+R.T)/2).min()<-1e-7:raise ValueError('Endpoint split exceeds existing measurement covariance')
        return H,R,value,flow

    def step(self,*args,**kwargs):
        c,v,f,s=super().step(*args,**kwargs)
        s['shared_image_endpoint_error_model']=dict(applied=True,endpoint_variance_px2=self.endpoint_variance,
            one_pair_floor_variance_px2=2*self.endpoint_variance,
            consecutive_pair_error_covariance_px2=-self.endpoint_variance,
            covariance_is_engineering_prior=True,extra_cycle_polarity_covariance_preserved=True,
            force_ownership_and_front_unchanged=True)
        return c,v,f,s


class EndpointNormalObserver(EndpointNoiseMixin,NormalResponseObserver):
    endpoint_base=12
    state_dim=16


class EndpointSeparatedObserver(EndpointNoiseMixin,SeparatedForceObserver):
    endpoint_base=14
    state_dim=18


class EndpointAdaptiveObserver(EndpointNormalObserver):
    alternative_type=EndpointSeparatedObserver
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.alternative=self.alternative_type(*args,**kwargs);self.credit=CausalResponseCredit()

    def step(self,t,body,u,*args,**kwargs):
        c,v,f,s=super().step(t,body,u,*args,**kwargs);self.alternative.two_point_init=self.two_point_init
        ca,va,fa,sa=self.alternative.step(t,body,u,*args,**kwargs)
        common=dict(t=t,body=body,structure=kwargs.get('structure') or {},motion=kwargs.get('motion') or {})
        base=dict(common,coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'))
        alt=dict(common,coefficient=ca,valid=va,force=fa,state=sa,normal_turn_forecast=sa.get('normal_turn_forecast'))
        result=self.credit.step(base,alt);result['state']['primary_internal_state_is_not_mixture_posterior']=True
        return result['coefficient'],result['valid'],result['force'],result['state']


class ProjectionQualifiedSeparatedObserver(EndpointSeparatedObserver):
    preserve_unmeasured_projection=True


class ProjectionQualifiedObserver(EndpointAdaptiveObserver):
    preserve_unmeasured_projection=True
    alternative_type=ProjectionQualifiedSeparatedObserver
