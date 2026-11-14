"""Gauge-aware gain uncertainty for current spatially validated ring inputs.

The existing1000px/s² acceleration prior defines a response uncertainty, not
an arbitrary fixed coefficient uncertainty. Broaden k only along the exact
gauge(d,k)->(d-u*delta,k+delta), preserving the current net force/covariance.
Constant input still has no gain information. The ordinary joint innovation
updates k only on retained, currently validated sources; no label/time/ID rule.
"""
import numpy as np
from layout_reference_observer import LayoutReferenceObserver


class PhaseGainObserver(LayoutReferenceObserver):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.gain_response_prior_initialized=False

    def _gain_learning_allowed(self,good,allow_gain_learning,history_length,excitation,membership_changed,input_meta):
        if (input_meta or {}).get('source')!='current_unordered_visible_ring_phase_sum':return False
        return bool(good and allow_gain_learning and self.previous_gain_qualified and not membership_changed)

    def _adjust_gain_update(self,K,residual,dt,excitation):
        candidate=self.x[self.gain_index]+float(K[self.gain_index]@residual)
        # Existing3000px/s² response limit; invariant under u->c*u,k->k/c.
        if not 0.<candidate or candidate*excitation>3000.:K[self.gain_index]=0.

    def step(self,t,body,u,*args,**kwargs):
        c,valid,f,s=super().step(t,body,u,*args,**kwargs)
        meta=kwargs.get('input_meta') or {}
        prior_applied=False
        if (not self.gain_response_prior_initialized and c is not None and f.get('active') and
                meta.get('source')=='current_unordered_visible_ring_phase_sum'):
            norm=float(np.linalg.norm(self.u))
            if norm>0:
                variance=(1000./norm)**2
                extra=max(0.,variance-self.P[self.gain_index,self.gain_index])
                gauge=np.zeros(self.state_dim);gauge[4:6]=-self.u;gauge[self.gain_index]=1.
                self.P+=extra*np.outer(gauge,gauge)
                self.gain_response_prior_initialized=True;prior_applied=True
                f['gain_sigma']=float(np.sqrt(self.P[self.gain_index,self.gain_index]))
        s.update(phase_gain_prior_broadened_without_net_force_change=prior_applied,
            phase_gain_response_prior_initialized=self.gain_response_prior_initialized,
            gain_gate_uses_current_validated_source_and_joint_information=True,
            gain_rule_is_input_unit_invariant=True)
        return c,valid,f,s
