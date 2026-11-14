"""Causal velocity-dependent unknown-force response, without measuring thrust.

Weak-perspective image model a = H*n + b - c*v. Projecting perpendicular to
the measured rotor-plane normal eliminates H, including hidden rotor thrust.
Only observed inter-frame displacement differences enter this regression.
c is an effective image-domain response, NOT a measured aerodynamic parameter.
"""
from collections import deque
import copy
import numpy as np
from scipy.special import log_ndtr
from axis_amplitude_bridge import current_axis_qualified


class TangentialResponseMemory:
    def __init__(self):self.previous=None;self.samples=deque();self.last_pair_t=None
    def observe(self,row):
        t=row['t'];motion=row.get('motion') or {};est=motion.get('estimate') or {};dt=motion.get('dt_s');structure=row['structure']
        good=bool(dt and motion.get('current_observation') and est.get('score',0)>0 and not est.get('peak_at_search_boundary') and est.get('current_displacement_qualified',True))
        info=dict(qualified=False,reason='insufficient_independent_motion_pairs',metric_drag_identified=False,
            current_or_future_trajectory_labels_used=False,hidden_rotor_force_cancelled_by_tangent_projection=True)
        if good:
            v=np.array([est['dx'],est['dy']])/dt
            R=np.asarray(est.get('displacement_covariance_px2',np.eye(2)*4.),float)/dt**2
            old=self.previous
            if old is not None and abs(t-old['t']-dt)<1e-6 and current_axis_qualified(old['structure']):
                n=np.asarray(old['structure'].get('axis_xy'),float)
                if n.shape==(2,) and np.isfinite(n).all() and np.linalg.norm(n)>1e-6:
                    tangent=np.array([-n[1],n[0]])/np.linalg.norm(n);interval=(dt+old['dt'])/2
                    if self.last_pair_t is None or t-self.last_pair_t>=2*min(dt,old['dt'])-1e-6:
                        acceleration=(v-old['v'])/interval;speed=(v+old['v'])/2
                        variance=float(tangent@(R+old['R'])@tangent)/interval**2
                        X=np.r_[tangent,-tangent@speed];y=float(tangent@acceleration)
                        self.samples.append((t,X,y,max(variance,1.)));self.last_pair_t=t
            self.previous=dict(t=t,v=v,R=R,dt=dt,structure=structure)
        else:self.previous=None
        while self.samples and t-self.samples[0][0]>3.:self.samples.popleft()
        if len(self.samples)<8 or self.samples[-1][0]-self.samples[0][0]<.4:return info
        prior=1/.35;mean=np.array([0.,0.,prior]);precision=np.diag([1e-6,1e-6,1/prior**2]);rhs=precision@mean
        for _,X,y,var in self.samples:precision+=np.outer(X,X)/var;rhs+=X*y/var
        covariance=np.linalg.inv(precision);beta=covariance@rhs;mu=float(beta[2]);sigma=float(np.sqrt(covariance[2,2]))
        z=mu/sigma;positive=mu+sigma*np.exp(-.5*z*z-.5*np.log(2*np.pi)-log_ndtr(z))
        information=float(np.clip(1-covariance[2,2]/prior**2,0.,1.))
        effective=(1-information)*prior+information*positive
        info.update(qualified=True,reason='past_tangential_motion_response',
            effective_response_s_inv=float(effective),unconstrained_response_mean_s_inv=mu,
            response_sigma_proxy_s_inv=sigma,positive_response_mean_s_inv=float(positive),
            information_fraction=information,unknown_constant_bias_px_s2=beta[:2].tolist(),
            samples=len(self.samples),window_s=3.,prior_response_s_inv=prior,
            uncertainty_calibrated=False,physical_assumptions='locally constant projected external bias and isotropic velocity response')
        return info


def adapt(row,info):
    r=copy.deepcopy(row);r['tangential_response_memory']=info
    if not info.get('qualified') or not row['motion'].get('current_observation'):return r
    def change(packet):
        f=packet['force']
        # Keep the existing short-gap policy; never make up a current image.
        if abs(float(f.get('acceleration_correlation_s',.35))-.35)<1e-8:
            f['acceleration_correlation_s']=1/info['effective_response_s_inv']
            f['future_correlation_from_tangential_response_proxy']=True
        for part in f.get('response_model_mixture',[]):change(part)
    change(r);return r
