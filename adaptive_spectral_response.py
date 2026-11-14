"""Causal innovation-calibrated process spectral density for the SAME observer.

Inspired by covariance matching / innovation-based adaptive estimation. This
small implementation is ours, not an author-code reproduction. It changes only
the next transition's Q; no labels, extra position branch or measurement gates.
The original .35s motion correlation supplies the adaptation timescale.
"""
import numpy as np
from adaptive_response_observer import AdaptiveResponseObserver
from separated_force_response import SeparatedForceObserver
from causal_response_credit import CausalResponseCredit


class SpectralMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.log_spectral_scale=0.;self.measurement_dimension=None;self.adaptation_t=None
    def _transition(self,dt):
        F,Q=super()._transition(dt)
        return F,Q*np.exp(self.log_spectral_scale)
    def _measurement(self,*args,**kwargs):
        H,R,z,used=super()._measurement(*args,**kwargs);self.measurement_dimension=len(z)
        return H,R,z,used
    def step(self,t,*args,**kwargs):
        self.measurement_dimension=None
        c,v,f,s=super().step(t,*args,**kwargs);nis=s.get('measurement_innovation_chi2');dim=self.measurement_dimension
        before=float(np.exp(self.log_spectral_scale))
        if nis is not None and dim and np.isfinite(nis):
            dt=0. if self.adaptation_t is None else max(0.,t-self.adaptation_t)
            eta=-np.expm1(-min(dt,.12)/self.correlation_s)
            # A normalized innovation has expectation1. Symmetric bounded
            # score prevents one segmentation outlier exploding Q.
            z=float(nis)/dim;score=(z-1)/(z+1)
            self.log_spectral_scale=float(np.clip(self.log_spectral_scale+eta*2*score,-np.log(1000),np.log(1000)))
            self.adaptation_t=float(t)
        s['causal_process_spectral_adaptation']=dict(applied_to_next_transition=True,scale_used=before,
            next_scale=float(np.exp(self.log_spectral_scale)),nis=nis,measurement_dimension=dim,
            same_native_observations_not_new_independent_evidence=True,calibrated_noise_truth=False)
        return c,v,f,s


class SpectralSeparated(SpectralMixin,SeparatedForceObserver):pass


class AdaptiveSpectralResponse(SpectralMixin,AdaptiveResponseObserver):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.alternative=SpectralSeparated(*args,**kwargs);self.credit=CausalResponseCredit()
