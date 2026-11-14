"""Relative rotor displacement as one measurement, preserving its correlation.

The two cloned previous-position states carry cross covariance. Integrated
matched centers are not repeatedly counted as independent absolute fixes.
Current qualified five-circle centers remain absolute observations. No new
trajectory branch or force law is added; hidden rotor force stays in d.
"""
import numpy as np
from rotor_force_observer import RotorForceObserver


class RelativeRotorForceObserver(RotorForceObserver):
    state_dim=9

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.clone_t=None;self.relative_used=False

    def _initialize(self,z):
        super()._initialize(z)
        self.x=np.r_[self.x,z,np.zeros(self.state_dim-9)];P=np.zeros((self.state_dim,self.state_dim));P[:7,:7]=self.P;self.P=P
        self._clone_position()

    def _clone_position(self):
        J=np.eye(self.state_dim);J[7:9]=0.;J[7,0]=1.;J[8,1]=1.
        self.x[7:9]=self.x[:2];self.P=J@self.P@J.T

    def _rebase_observation_reference(self,shift):
        self.x[:2]+=shift;self.x[7:9]+=shift
        J=np.zeros((self.state_dim,2));J[:2]=np.eye(2);J[7:9]=np.eye(2)
        self.P+=36.*J@J.T

    def _measurement(self,z,body,motion,lag):
        self.relative_used=False
        shift=(motion or {}).get('estimate') or {};dt=(motion or {}).get('dt_s')
        if (not lag and body.get('measurement_source') in ('filtered_native_translation','expanded_rotor_support_reference') and
                dt is not None and 0<dt<=.12 and self.clone_t is not None and
                abs((self.t-self.clone_t)-dt)<1e-6 and shift.get('score',0)>0 and
                not shift.get('peak_at_search_boundary',False) and shift.get('current_displacement_qualified',True)):
            delta=np.array([shift['dx'],shift['dy']],float)
            if np.isfinite(delta).all():
                H=np.zeros((2,self.state_dim));H[:,:2]=np.eye(2);H[:,7:9]=-np.eye(2)
                self.relative_used=True
                # Same declared2px displacement prior as the existing optional
                # flow observation, without inventing independent body fixes.
                covariance=np.asarray(shift.get('displacement_covariance_px2',np.eye(2)*4.),float)
                if covariance.shape!=(2,2) or not np.isfinite(covariance).all():raise ValueError('Finite2x2 displacement covariance required')
                covariance=(covariance+covariance.T)*.5
                # Keep the original2px floor, retaining anisotropic cycle
                # uncertainty instead of silently ignoring available quality.
                values,vectors=np.linalg.eigh(covariance);covariance=(vectors*np.maximum(values,4.))@vectors.T
                return H,covariance,delta,False
        return super()._measurement(z,body,None,lag)

    def step(self,t,body,u,*args,**kwargs):
        # _measurement is skipped for a provisional/prediction-only reference.
        # Do not carry the preceding frame's successful-update flag forward.
        self.relative_used=False
        c,valid,f,s=super().step(t,body,u,*args,**kwargs)
        s.update(native_relative_displacement_observed=self.relative_used,
            previous_position_clone_preserves_cross_covariance=True,
            integrated_rotor_center_is_independent_absolute_fix=False)
        prediction_reference=bool(body.get('motion_reference_advanced_without_measurement'))
        s['prediction_only_image_reference_cloned']=bool(c is not None and prediction_reference)
        if c is not None and body.get('detected') and (not body.get('center_provisional',False) or prediction_reference):
            self._clone_position();self.clone_t=float(t)
        return c,valid,f,s
