"""Unit-invariant input excitation, separated from constant unknown forces.

The old absolute .08 input gate changes its answer if the arbitrary reference
frequency changes. Project u(t) away from a constant 2D external force and a
constant unknown normal response h*n(t), then compare residual input energy
to its OWN uncertainty. This qualifies input excitation, not calibrated thrust.
"""
from collections import deque
import numpy as np
from adaptive_response_observer import AdaptiveResponseObserver
from separated_force_response import SeparatedForceObserver
from axis_amplitude_bridge import current_axis_qualified


def excitation(samples):
    if len(samples)<8:return dict(qualified=False,reason='need_eight_matured_current_inputs',rows=len(samples),reliability=0.)
    n=np.asarray([r[2] for r in samples]);u=np.asarray([r[1] for r in samples]);sigma=np.repeat([r[3] for r in samples],2)
    B=np.zeros((2*len(samples),3));B[::2,0]=1.;B[1::2,1]=1.;B[:,2]=n.reshape(-1)
    A=B/sigma[:,None];y=u.reshape(-1)/sigma
    q=y-A@np.linalg.lstsq(A,y,rcond=None)[0]
    rank=int(np.linalg.matrix_rank(A));dof=len(y)-rank;energy=float(q@q)
    credit=max(0.,1-dof/max(energy,1e-12))
    return dict(qualified=credit>0,reason='input_excitation_resolved' if credit>0 else 'input_variation_not_resolved_above_own_noise',
        rows=len(samples),nuisance_rank=rank,residual_degrees_of_freedom=dof,
        normalized_input_energy=energy,input_noise_ratio_proxy=float(np.sqrt(energy/max(dof,1))),
        reliability=float(credit),input_units_change_decision=False,
        constant_external_and_normal_response_removed=True,metric_force_identified=False,
        input_covariance_independence_and_calibration_not_proved=True)


class InputGainMixin:
    def __init__(self,*args,**kwargs):
        self.gain_samples=deque();self.gain_keys=();self.input_gain_diagnostic=dict(qualified=False,reliability=0.)
        super().__init__(*args,**kwargs)

    def _gain_learning_allowed(self,good,allow_gain_learning,history_length,excitation_rms,membership_changed,input_meta):
        info=excitation(list(self.gain_samples));self.input_gain_diagnostic=info
        return bool(good and allow_gain_learning and self.previous_gain_qualified and not membership_changed and info['qualified'])

    def _adjust_gain_update(self,K,residual,dt,excitation_rms):
        K[self.gain_index]*=-np.expm1(-dt/1.)*self.input_gain_diagnostic.get('reliability',0.)
        candidate=self.x[self.gain_index]+float(K[self.gain_index]@residual)
        # Same25x initial-gain bound as the old200->5000 prior, but now the
        # bound follows a pure change of input units as well.
        if not 0.<candidate<25*self.default_gain:K[self.gain_index]=0.

    def step(self,t,body,u,*args,**kwargs):
        result=super().step(t,body,u,*args,**kwargs);c,v,f,s=result
        meta=kwargs.get('input_meta') or {};keys=tuple(sorted(f.get('observed_components',{})))
        st=kwargs.get('structure') or {};n=st.get('axis_xy');sigma=float(meta.get('input_sigma_proxy',0.))
        if keys!=self.gain_keys:self.gain_samples.clear()
        self.gain_keys=keys
        allowed=kwargs.get('allow_gain_learning',True)
        if f.get('active') and allowed and sigma>0 and n is not None and current_axis_qualified(st):
            # Append AFTER the position update. A new input is allowed to help
            # identify K only when a later position observation has matured.
            self.gain_samples.append((t,np.asarray(u,float).copy(),np.asarray(n,float).copy(),sigma))
        else:self.gain_samples.clear()
        while self.gain_samples and t-self.gain_samples[0][0]>3.:self.gain_samples.popleft()
        f['input_gain_qualification']=dict(self.input_gain_diagnostic,history_is_past_inputs_before_current_measurement=True)
        return c,v,f,s


class ObservableSeparatedForce(InputGainMixin,SeparatedForceObserver):pass


class ObservableGainObserver(InputGainMixin,AdaptiveResponseObserver):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.alternative=ObservableSeparatedForce(*args,**kwargs)
