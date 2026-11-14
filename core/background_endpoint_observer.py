"""Shared background-image errors in stabilized rotor displacement factors.

delta_world=delta_object+(e_fg_now-e_fg_old)-(e_bg_now-e_bg_old)+residual.
Keep the original foreground model and background4px² floor. Split only that
background floor into two2px² image endpoints. This is not independent repeated
camera acceleration noise, nor a pixel-count precision multiplication.
"""
import numpy as np
from equilibrium_gain_prior import CovarianceOnlyObserver,CovarianceOnlySeparated

class BackgroundEndpointMixin:
    def _initialize(self,z):
        super()._initialize(z);a=self.background_endpoint_base
        self.P[a:a+4,a:a+4]=np.eye(4)*2.

    def _transition(self,dt):
        F,Q=super()._transition(dt);a=self.background_endpoint_base
        F[a+2:a+4]=0.;Q[a+2:a+4]=0.;Q[:,a+2:a+4]=0.
        Q[a+2:a+4,a+2:a+4]=np.eye(2)*2.
        return F,Q

    def _clone_position(self):
        super()._clone_position();a=self.background_endpoint_base
        if not np.any(self.P[a+2:a+4,a+2:a+4]):self.P[a+2:a+4,a+2:a+4]=np.eye(2)*2.
        J=np.eye(self.state_dim);J[a:a+2]=0.;J[a:a+2,a+2:a+4]=np.eye(2)
        self.x[a:a+2]=self.x[a+2:a+4];self.P=J@self.P@J.T

    def _measurement(self,z,body,motion,lag):
        H,R,value,flow=super()._measurement(z,body,motion,lag);info=(motion or {}).get('background_reference_interval') or {}
        self.background_endpoint_used=bool(self.relative_used and info.get('fully_observed') and info.get('shared_endpoint_floor_enabled'))
        if self.background_endpoint_used:
            a=self.background_endpoint_base;H=H.copy();R=R.copy()
            H[:2,a:a+2]=np.eye(2);H[:2,a+2:a+4]=-np.eye(2)
            R[:2,:2]-=np.eye(2)*4.
            assert np.linalg.eigvalsh((R+R.T)/2).min()>-1e-7,dict(
                error='Background floor was not actually present in R',t=self.t,R=R.tolist(),
                shift_covariance=(motion.get('estimate') or {}).get('displacement_covariance_px2'),
                added_bg=(motion.get('estimate') or {}).get('background_reference_added_covariance_px2'),interval=info)
        return H,R,value,flow

    def step(self,*args,**kwargs):
        self.background_endpoint_used=False;c,v,f,s=super().step(*args,**kwargs)
        s['shared_background_endpoint_error']=dict(applied=bool(self.background_endpoint_used),
            one_pair_floor_px2=4.,endpoint_floor_px2=2.,adjacent_pair_error_covariance_px2=-2.,
            extra_covariance_residual_retained=True,unobserved_background_gaps_not_given_endpoint_credit=True,
            camera_uncertainty_is_engineering_prior=True)
        return c,v,f,s

class BackgroundEndpointSeparated(BackgroundEndpointMixin,CovarianceOnlySeparated):
    background_endpoint_base=18;state_dim=22

class BackgroundEndpointObserver(BackgroundEndpointMixin,CovarianceOnlyObserver):
    background_endpoint_base=16;state_dim=20;alternative_type=BackgroundEndpointSeparated
