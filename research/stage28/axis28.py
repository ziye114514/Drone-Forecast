"""Isolate angular propagation and signed-canonical measurement fusion."""
from pathlib import Path
import importlib.util,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage27'),str(ROOT/'src')]
import physics27 as prior
from event_drone_forecast.paper_physics import ThrustAxisFilter,unit

METHODS=['original','no_rate','canonical_static','instantaneous']
PARAMS=dict(prior.PARAMS,axis_variants=METHODS,normal_process_sigma=prior.PaperConfig().normal_process_sigma,
    canonical_assumption='Known image-up half-plane and weak-perspective nonnegative depth component, identical to original per-frame weak_normal prior. Not a general camera-roll/inverted-flight solution.',
    static_prior='No unobserved normal rotation between measurements or in future rollout; same original measurement consensus, same covariance constants.')


class NoRate(ThrustAxisFilter):
    def step(self,*args,**kwargs):
        self.omega[:]=0.;n=super().step(*args,**kwargs);self.omega[:]=0.;return n


class CanonicalStatic:
    def __init__(self,cfg,instantaneous=False):
        self.cfg=cfg;self.instantaneous=instantaneous;self.n=None;self.omega=np.zeros(3);self.P=np.eye(3)*.2
        self.t=None;self.last_measurement_t=None;self.quality=0.

    def step(self,t,rotors,differential=0.,enhanced=True):
        dt=0. if self.t is None else max(0.,t-self.t);self.t=t
        self.P+=np.eye(3)*self.cfg.normal_process_sigma**2*dt
        # Execute original circle/ellipse-to-normal and multi-rotor consensus from
        # a fresh instance, hence no temporal antipodal flip or inferred rotation.
        measurement=ThrustAxisFilter(self.cfg);n=measurement.step(t,rotors,0.,enhanced)
        if n is None:return self.n
        self.last_measurement_t=t;self.quality=measurement.quality
        if self.n is None or self.instantaneous:self.n=n.copy()
        else:
            R=np.eye(3)*(.025/max(self.quality,.1))**2;gain=np.linalg.solve(self.P+R,self.P).T
            self.n=unit(self.n+gain@(n-self.n));self.P=(np.eye(3)-gain)@self.P
        return self.n.copy()


class Predictor(prior.Predictor):
    def __init__(self,method='canonical_static'):
        if method not in METHODS:raise ValueError(method)
        self.method=method;super().__init__()

    def reset_features(self):
        super().reset_features()
        if self.method=='no_rate':self.axis=NoRate(self.cfg)
        elif self.method in ['canonical_static','instantaneous']:
            self.axis=CanonicalStatic(self.cfg,self.method=='instantaneous')
