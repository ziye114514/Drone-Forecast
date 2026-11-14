"""Causal force increments: avoid adding already observed steady thrust twice.

Frequency candidates are tracked continuously with retained half/double
ambiguity. There is still no mechanical RPM identification. The only learned
coefficient is a nonnegative effective gain on force-feature *changes*.
"""
from pathlib import Path
from collections import deque
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage69'),str(ROOT/'research/stage68')]
from force69 import Direction,prior
from activity68 import Reference
ADDED=['increment_density_constant','increment_density','increment_periodic_constant','increment_periodic']
PARAMS=dict(ema_time_s=.2,frequency_fraction_step_max=.2,frequency_expiry_s=.16,
    harmonic_hypotheses=[.5,1.,2.],gain_bounds=[0.,5000.],initial_gain=0.,initial_covariance=.5,
    delay_s=.2,feedback_expiry_s=.26,warmup_s=.6,forgetting=.999,
    maximum_feature_innovation=2.,minimum_feature_excitation=.035,
    scope='Nonnegative image-domain gain on current feature minus past EMA. '
          'History already includes previously observed acceleration; only '
          'input innovation is added. Causal frequency continuity is a prior '
          'that can be wrong, not a resolution of blade-count/harmonic ambiguity.')


class ContinuousFrequency:
    def __init__(self):self.hz=None;self.t=None
    def step(self,t,activity):
        candidates=[c for c in activity.get('spectral_candidates',[]) if c['qualified']]
        if not candidates:return None,dict(current=False,reason='no_qualified_tone',hz=self.hz)
        if self.t is None or t-self.t>.16:
            strongest=max(candidates,key=lambda c:c['tone_fraction']*c['phase_coherence'])
            self.hz=strongest['frequency_hz'];self.t=t
            return self.hz**2,dict(current=True,hz=self.hz,reason='new_relative_frequency_reference',harmonic_ambiguous=True)
        usable=[]
        for c in candidates:
            for multiplier in [.5,1.,2.]:
                f=c['frequency_hz']*multiplier
                if 60<=f<=1200 and abs(f/self.hz-1)<=.2:
                    weight=c['tone_fraction']*c['phase_coherence']*np.exp(-.5*((f/self.hz-1)/.1)**2)
                    usable.append((f,weight,multiplier))
        if not usable:return None,dict(current=False,hz=self.hz,reason='frequency_jump_rejected',harmonic_ambiguous=True)
        values=np.array([c[0] for c in usable]);weights=np.array([c[1] for c in usable]);order=np.argsort(values)
        index=order[np.searchsorted(np.cumsum(weights[order]),weights.sum()/2)]
        self.hz=float(values[index]);self.t=t
        return self.hz**2,dict(current=True,hz=self.hz,reason='continuous_relative_tone',
                             retained_multiplier=usable[index][2],harmonic_ambiguous=True)


class Gain:
    def __init__(self):self.theta=0.;self.P=.5;self.mean=None;self.last_t=None;self.started=None;self.queue=deque();self.updates=0
    def step(self,t,base,n,s,active,current,feedback,center):
        if self.started is None:self.started=t
        delta=None
        if active:
            force=np.asarray(n)*s
            if self.mean is None or self.last_t is None or t-self.last_t>.16:self.mean=force.copy()
            delta=force-self.mean
            length=np.linalg.norm(delta)
            if length>2:delta*=2/length
            dt=.02 if self.last_t is None else t-self.last_t
            self.mean+=(-np.expm1(-dt/.2))*(force-self.mean);self.last_t=t
        while self.queue and t-self.queue[0][0]>.26000001:self.queue.popleft()
        if feedback:
            while self.queue and t-self.queue[0][0]>=.2-1e-8:
                old_t,old_base,X=self.queue.popleft();dt=t-old_t
                target=2*(np.asarray(center)-np.asarray(prior.evaluate(old_base,dt)))/dt**2/200.
                error=np.clip(target-X*self.theta,-2.,2.)
                K=self.P*X/(.5+self.P*(X@X));self.theta=float(np.clip(self.theta+K@error,0.,25.))
                self.P=float(np.clip((self.P-self.P*(K@X))/.999,1e-6,100));self.updates+=1
        launch=bool(active and current and feedback and t-self.started>=.6 and np.linalg.norm(delta)>=.035)
        if launch:self.queue.append((t,copy.deepcopy(base),delta.copy()))
        a=200.*self.theta*delta if active else np.zeros(2)
        return a,dict(active=active,launch=launch,updates=self.updates,gain_px_s2=200*self.theta,
            delta=None if delta is None else delta.tolist(),ratio=s,covariance=self.P)


class IncrementForecast:
    def __init__(self):self.reset()
    def reset(self):
        self.direction=Direction();self.frequency=ContinuousFrequency();self.refs={k:Reference() for k in ['density','periodic']}
        self.models={m:Gain() for m in ADDED};self.last_seen=None
    def step(self,original,obs,body,activity,rotor_quality):
        q=copy.deepcopy(original);t=q['t']
        if self.last_seen is not None and t-self.last_seen>.5:self.reset()
        if obs.get('detected'):self.last_seen=t
        n,nd=self.direction.step(t,q,body,rotor_quality);frequency,fd=self.frequency.step(t,activity)
        density=activity.get('density');values=dict(density=density**2 if density is not None and density>0 else None,periodic=frequency)
        features={k:self.refs[k].step(t,v) for k,v in values.items()};models={}
        for name,model in self.models.items():
            kind='density' if 'density' in name else 'periodic';f=features[kind]
            s=1. if name.endswith('_constant') else f['ratio']
            active=bool(q['valid'] and n is not None and f['active'])
            feedback=bool(q['valid'] and obs.get('detected') and obs.get('confidence',0)>.5)
            current=bool(nd['current'] and nd['quality']>.45 and f['current'])
            if q['valid']:
                base=q['coefficients']['trusted_acc'];a,diag=model.step(t,base,n,s,active,current,feedback,obs.get('center'))
                coefficient=copy.deepcopy(base);coefficient['acceleration']=(np.asarray(coefficient['acceleration'])+a).tolist()
                q['coefficients'][name]=coefficient;q['predictions'][name]={str(h):prior.evaluate(coefficient,h) for h in prior.PARAMS['horizons']}
            else:diag=dict(active=False,launch=False,updates=model.updates,gain_px_s2=200*model.theta,delta=None,ratio=s,covariance=model.P)
            models[name]=diag
        q['increment_diagnostic']=dict(direction=nd,frequency=fd,features=features,models=models)
        return q


def controls():
    model=Gain();base=dict(position=[0.,0.],velocity=[0.,0.],acceleration=[0.,0.]);n=np.array([.6,-.8])
    for j in range(100):
        a,d=model.step(j*.02,base,n,1.,True,True,True,[0.,0.])
        np.testing.assert_array_equal(a,[0.,0.]);assert not d['launch']
    assert model.updates==0
    tracker=ContinuousFrequency()
    def activity(f):return dict(spectral_candidates=[dict(qualified=True,frequency_hz=f,tone_fraction=.5,phase_coherence=.9)])
    first,_=tracker.step(0.,activity(200.));second,diag=tracker.step(.02,activity(400.))
    assert first==second and diag['retained_multiplier']==.5
    return dict(steady_force_innovation_zero=True,no_steady_double_count=True,
                harmonic_continuity_prior_checked=True,mechanical_RPM_not_identified=True)
