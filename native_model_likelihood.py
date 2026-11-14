"""Current-factor dynamic model averaging of two existing rotor-force models.

Inspired by Markov switching filters, not a full IMM: model posteriors are
not mixed back into each other's states. Native factors are conditional on
shared endpoint states. Their engineering covariances are not calibrated.
No long-horizon future outcome selects the current model probability.
"""
from collections import deque
import numpy as np
from positive_normal_startup import PositiveStartupObserver,PositiveSeparatedObserver
from equilibrium_gain_prior import EquilibriumGainObserver,EquilibriumSeparatedObserver
from shared_endpoint_observer import EndpointNormalObserver
from causal_response_credit import CausalResponseCredit


class NativeLikelihoodMixin:
    def _measurement(self,z,body,motion,lag):
        H,R,value,flow=super()._measurement(z,body,motion,lag)
        if lag:raise ValueError('Current-factor evidence requires current observations')
        if self.relative_used:
            e=(motion or {}).get('estimate') or {};C=np.asarray(e.get('displacement_covariance_px2',np.eye(2)*4.),float)
            C=(C+C.T)/2;w,V=np.linalg.eigh(C);C=(V*np.maximum(w,4.))@V.T
            Re=C-4.*np.eye(2);kind='native_relative'
        else:Re=np.eye(2)*34.;kind='native_absolute'
        # Use the same marginal native factor even when the other model's
        # uncertain layout gate differs. No comparison of differently sized
        # layout stacks or extra independent image observations is made.
        h=H[:2];residual=value[:2]-h@self.x;S=h@self.P@h.T+Re;S=(S+S.T)/2
        sign,logdet=np.linalg.slogdet(S)
        if sign<=0:raise ValueError('Positive conditional native covariance required')
        chi=float(residual@np.linalg.solve(S,residual))
        self.native_likelihood=dict(log_likelihood=float(-.5*(chi+logdet+2*np.log(2*np.pi))),
            chi2=chi,logdet=float(logdet),factor=kind,conditional_on_shared_endpoint_state=True,
            engineering_uncertainty_not_calibrated=True)
        return H,R,value,flow


class FixedCreditMemory:
    def __init__(self):self.weight=0.;self.rows=[];self.pending=deque()
    def step(self,*args,**kwargs):pass
    def rebase(self,*args,**kwargs):pass


class PredictiveModelCredit(CausalResponseCredit):
    def __init__(self):
        super().__init__();self.memory=FixedCreditMemory();self.probability=np.array([1.,0.]);self.t=None

    def step(self,baseline,alternative):
        t=float(baseline['t']);dt=0. if self.t is None else t-self.t;self.t=t
        if dt<0:raise ValueError('Increasing source time required')
        # Existing force-memory timescale; no fitted mode-switch parameter.
        stay=.5*(1+np.exp(-dt/.35));prior=np.array([[stay,1-stay],[1-stay,stay]])@self.probability
        a=baseline['state'].get('native_predictive_likelihood');b=alternative['state'].get('native_predictive_likelihood');used=False
        if a and b and a['factor']==b['factor']:
            logp=np.log(np.maximum(prior,1e-300))+np.array([a['log_likelihood'],b['log_likelihood']]);p=np.exp(logp-np.max(logp));self.probability=p/p.sum();used=True
        else:self.probability=prior
        self.memory.weight=float(self.probability[1]);self.last_origin=t
        r=super().step(baseline,alternative)
        r['state']['predictive_model_credit']=dict(probabilities=self.probability.tolist(),prior=prior.tolist(),
            current_native_likelihood_used=used,current_factors=[a,b],transition_correlation_s=.35,
            future_outcomes_used=False,mode_state_interaction=False,not_full_IMM=True,
            plane_qualification_unchanged=True,uncertainty_not_calibrated=True)
        return r


def diagnostics(observer,s):
    s['native_predictive_likelihood']=getattr(observer,'native_likelihood',None)
    if hasattr(observer,'equilibrium_gain_prior'):s['equilibrium_gain_prior']=observer.equilibrium_gain_prior
    return s


class LikelihoodSeparated(NativeLikelihoodMixin,PositiveSeparatedObserver):
    def step(self,*args,**kwargs):
        self.native_likelihood=None;c,v,f,s=super().step(*args,**kwargs);return c,v,f,diagnostics(self,s)


class LikelihoodEquilibriumSeparated(NativeLikelihoodMixin,EquilibriumSeparatedObserver):
    def step(self,*args,**kwargs):
        self.native_likelihood=None;c,v,f,s=super().step(*args,**kwargs);return c,v,f,diagnostics(self,s)


class PredictiveCreditObserver(NativeLikelihoodMixin,PositiveStartupObserver):
    alternative_type=LikelihoodSeparated
    use_predictive_credit=True
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.use_predictive_credit:self.credit=PredictiveModelCredit()
    def step(self,t,body,u,*args,**kwargs):
        self.native_likelihood=None
        c,v,f,s=EndpointNormalObserver.step(self,t,body,u,*args,**kwargs);s=diagnostics(self,s)
        self.alternative.two_point_init=self.two_point_init
        ca,va,fa,sa=self.alternative.step(t,body,u,*args,**kwargs)
        common=dict(t=t,body=body,structure=kwargs.get('structure') or {},motion=kwargs.get('motion') or {})
        base=dict(common,coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'))
        alt=dict(common,coefficient=ca,valid=va,force=fa,state=sa,normal_turn_forecast=sa.get('normal_turn_forecast'))
        result=self.credit.step(base,alt);result['state']['primary_internal_state_is_not_mixture_posterior']=True
        return result['coefficient'],result['valid'],result['force'],result['state']


class PredictiveEquilibriumObserver(PredictiveCreditObserver,EquilibriumGainObserver):
    alternative_type=LikelihoodEquilibriumSeparated


class PreservedCreditControl(PredictiveCreditObserver):
    use_predictive_credit=False
