"""Tie the uncertain f^2 gain prior to the uncertain normal-drive scale.

Four exchangeable rotor frequencies are ONLY an initialization prior. No
hidden rotor frequency is put in the observed sum. New evidence still owns
the observed/unknown partition, and this is not a thrust calibration.
"""
import numpy as np
from positive_normal_startup import PositiveStartupObserver,PositiveSeparatedObserver


def anchor_metadata(meta,selected_frequency_hz):
    info=dict(meta);source=meta.get('source');den=float(meta.get('fixed_denominator_f2',360000.))
    anchor=None
    if source=='ambiguous_pooled_f2_change_proxy' and selected_frequency_hz is not None and meta.get('relative_f2') is not None:
        anchor=float(selected_frequency_hz)**2-float(meta['relative_f2'])*den
    elif source=='current_visible_slot_f2_sum':
        frequencies=list(meta.get('current_individual_frequency_hz',{}).values())
        if frequencies:anchor=float(np.mean(np.square(frequencies)))
    elif source=='anonymous_current_disjoint_partial_f2_sum':
        frequencies=meta.get('anonymous_current_region_frequencies_hz') or []
        if frequencies:anchor=float(np.mean(np.square(frequencies)))
    if anchor is not None and np.isfinite(anchor) and anchor>0:
        info['normal_gain_prior_anchor']=dict(frequency_squared_hz2=anchor,reference_hz=float(np.sqrt(den/4)),
            four_exchangeable_frequencies_are_prior_not_hidden_measurements=True)
    return info


class EquilibriumGainMixin:
    matched_second_moment_mean_control=False
    def _initialize(self,z):
        super()._initialize(z)
        self.initial_normal_mean=float(self.x[self.response_index])
        self.initial_normal_sigma=float(np.sqrt(self.P[self.response_index,self.response_index]))
        self.equilibrium_gain_prior=None

    def _gain_learning_allowed(self,good,allow_gain_learning,history_length,excitation,membership_changed,input_meta):
        data=(input_meta or {}).get('normal_gain_prior_anchor')
        if good and self.equilibrium_gain_prior is None and data:
            factor=float(data['reference_hz'])**2/float(data['frequency_squared_hz2'])
            mean=self.initial_normal_mean*factor;sigma=self.initial_normal_sigma*factor
            old=float(self.x[self.gain_index])
            if old<=0:raise ValueError('Positive startup gain required')
            ablated=False
            if self.matched_second_moment_mean_control and sigma*sigma+mean*mean>=old*old:
                sigma=float(np.sqrt(sigma*sigma+mean*mean-old*old));mean=old;ablated=True
            scale=mean/old;J=np.eye(self.state_dim);J[self.gain_index,self.gain_index]=scale
            J[4:6,self.gain_index]=(1-scale)*self.u
            direction=None
            if self.normal is not None and float(self.normal@self.normal)>1e-8:
                direction=self.normal/float(self.normal@self.normal)
                J[self.response_index,self.gain_index]=(1-scale)*float(direction@self.u)
            self.x=J@self.x;self.P=J@self.P@J.T
            # Retain a full old1000px/s² normal-scale uncertainty, including
            # compensating d/h covariance if a prior input is already present.
            extra=max(0.,sigma*sigma-self.P[self.gain_index,self.gain_index]);b=np.zeros(self.state_dim)
            b[self.gain_index]=1.;b[4:6]=-self.u
            if direction is not None:b[self.response_index]=-float(direction@self.u)
            self.P+=extra*np.outer(b,b)
            self.equilibrium_gain_prior=dict(applied=True,mean=mean,sigma=sigma,
                anchor=data,present_net_acceleration_preserved=True,
                mean_ablation_with_matched_second_moment=ablated,
                initial_normal_scale_is_prior=True,metric_thrust_measured=False)
        return super()._gain_learning_allowed(good,allow_gain_learning,history_length,excitation,membership_changed,input_meta)

    def step(self,*args,**kwargs):
        c,v,f,s=super().step(*args,**kwargs)
        s['equilibrium_gain_prior']=getattr(self,'equilibrium_gain_prior',None)
        return c,v,f,s


class EquilibriumSeparatedObserver(EquilibriumGainMixin,PositiveSeparatedObserver):pass

class EquilibriumGainObserver(EquilibriumGainMixin,PositiveStartupObserver):
    alternative_type=EquilibriumSeparatedObserver

class CovarianceOnlySeparated(EquilibriumSeparatedObserver):matched_second_moment_mean_control=True
class CovarianceOnlyObserver(EquilibriumGainObserver):
    matched_second_moment_mean_control=True
    alternative_type=CovarianceOnlySeparated


class NominalFrequencyBank:
    def __init__(self,base):self.base=base
    @property
    def accepts_anonymous_current_conics(self):return bool(getattr(self.base,'accepts_anonymous_current_conics',False))
    def step(self,t,structure,rotors,rates,pooled):
        u,q,meta=self.base.step(t,structure,rotors,rates,pooled)
        return u,q,anchor_metadata(meta,pooled.get('frequency_hz') if pooled.get('qualified') else None)
