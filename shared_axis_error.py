"""Retain the shared angular endpoint in successive normal increments.

d=e+h*n. New normal transport uses h*(n_new-n_old), hence its observation
error contains BOTH angular endpoints. Keep the old endpoint as a conditional
state, remove its contribution, and install one independent current endpoint.
The old5-degree prior, force means and Q are retained. This is first-order
Gaussian transport, not a calibrated attitude or Newton-force measurement.
"""
import numpy as np
from equilibrium_gain_prior import CovarianceOnlyObserver,CovarianceOnlySeparated


class SharedAxisMixin:
    def _initialize(self,z):
        super()._initialize(z);self.axis_error_active=False

    def _transport_normal(self,n,projection_scale):
        j=self.axis_error_index;hidx=self.response_index
        if self.axis_error_active:
            tangent=np.array([-self.normal[1],self.normal[0]])
            h=float(self.x[hidx]);error=float(self.x[j]);J=np.eye(self.state_dim)
            self.x[4:6]-=h*tangent*error
            J[4:6,j]-=h*tangent;J[4:6,hidx]-=tangent*error
            J[j]=0.;self.x[j]=0.;self.P=J@self.P@J.T
        else:
            self.x[j]=0.;self.P[j]=0.;self.P[:,j]=0.
        used=super()._transport_normal(n,projection_scale)
        # Parent has installed exactly E[h²]*sigma²*t*t^T in d. Keep its
        # cross covariance with this observed orientation, rather than
        # counting it as a fresh physical random acceleration every frame.
        variance=np.deg2rad(5)**2;tangent=np.array([-n[1],n[0]])
        self.P[j,j]=variance;self.P[4:6,j]=self.x[hidx]*variance*tangent
        self.P[j,4:6]=self.P[4:6,j];self.axis_error_active=True
        self.P=(self.P+self.P.T)/2
        return used

    def step(self,*args,**kwargs):
        was_two=self.two_point_initialized
        c,v,f,s=super().step(*args,**kwargs)
        if (s.get('normal_reference_reanchored_without_impulse') or
                (self.two_point_initialized and not was_two)):
            j=self.axis_error_index;self.x[j]=0.;self.P[j]=0.;self.P[:,j]=0.;self.axis_error_active=False
        s['shared_normal_observation_error']=dict(active=bool(getattr(self,'axis_error_active',False)),
            angular_sigma_deg=5.,adjacent_normal_endpoint_correlation_preserved=True,
            current_native_geometry_and_frequencies_unchanged=True,uncertainty_is_engineering_prior=True)
        return c,v,f,s


class SharedAxisSeparated(SharedAxisMixin,CovarianceOnlySeparated):
    axis_error_index=18;state_dim=19


class SharedAxisObserver(SharedAxisMixin,CovarianceOnlyObserver):
    axis_error_index=16;state_dim=17;alternative_type=SharedAxisSeparated
