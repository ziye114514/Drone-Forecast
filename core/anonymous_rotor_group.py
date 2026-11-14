"""Current disjoint periodic conics need not have four-template motor IDs.

Only a visible partial input proxy is returned, never total thrust or RPM.
Disjoint circumscribed discs are a deliberately conservative independence
condition. Current native conic/rate qualification is reused unchanged.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from rotor_thrust_bank import RotorThrustBankProxy
from research.stage146.branches146 import plane_axis


class AnonymousRotorGroup:
    ttl_s=.12
    accepts_anonymous_current_conics=True
    def __init__(self,base=None):
        self.base=base or RotorThrustBankProxy(vector_pool=True)
        self.previous=None;self.epoch=0;self.active=False

    def step(self,t,structure,rotors,rates,pooled):
        named=structure.get('mode') in ('four','three','partial_two')
        original=self.base.step(t,structure,rotors,rates if named else [],pooled)
        if original[2]['source']=='current_visible_slot_f2_sum':
            self.previous=None;self.active=False;return original
        selected=[(r,q) for r,q in zip(rotors,rates) if q.get('valid') and r.get('current_observation',True)
            and not r.get('fov_clipped') and r.get('ratio',0)>.16
            and (r.get('mode')=='ellipse' or r.get('arc_center_qualified') or r.get('curvature_validated'))]
        if not 2<=len(selected)<=4:
            self.previous=None;self.active=False;return original
        centers=np.array([[r['cx'],r['cy']] for r,q in selected]);sizes=np.array([r['major'] for r,q in selected]);frequencies=np.array([q['frequency_hz'] for r,q in selected])
        # No overlapping regional evidence may be counted twice as two motors.
        if any(np.linalg.norm(centers[i]-centers[j])<=.5*(sizes[i]+sizes[j]) for i in range(len(selected)) for j in range(i)):
            self.previous=None;self.active=False;return original
        n,agreement=plane_axis([r for r,q in selected])
        if agreement<.85 or not np.isfinite(n).all() or np.linalg.norm(n)<1e-6:
            self.previous=None;self.active=False;return original
        if n[1]>0:n=-n
        relative=centers-centers.mean(0);old=self.previous;stable=False
        if old is not None and 0<t-old['t']<self.ttl_s and len(old['f'])==len(frequencies):
            ii,jj=linear_sum_assignment(np.linalg.norm(old['relative'][:,None]-relative[None],axis=2))
            spatial=np.linalg.norm(old['relative'][ii]-relative[jj],axis=1)
            stable=bool(np.all(spatial<=np.maximum(4.,.15*sizes[jj])) and
                np.all(abs(np.log(frequencies[jj]/old['f'][ii]))<RotorThrustBankProxy.current_change_limit))
        self.previous=dict(t=float(t),relative=relative.copy(),f=frequencies.copy())
        if not stable:self.active=False;return original
        if not self.active:self.epoch+=1
        self.active=True;denominator=4*RotorThrustBankProxy.reference_hz**2
        value=float(np.sum(frequencies**2)/denominator);component=value*np.asarray(n)
        sigma=float(np.sqrt(np.sum((.04*frequencies**2)**2))/denominator)
        info=dict(original[2],source='anonymous_current_disjoint_partial_f2_sum',relative_f2=value,
            observed_components={f'anonymous_disjoint_group:{self.epoch}':component.tolist()},
            input_sigma_proxy=sigma,coherent=True,learning_input_current=True,rotor_bank_used=False,
            reason='current_disjoint_conics_and_rates_temporally_supported',mixed_spectrum=False,
            current_model_ids=[],hidden_model_ids=[],current_individual_frequency_hz={},
            anonymous_current_region_frequencies_hz=frequencies.tolist(),current_visible_region_count=len(selected),
            group_has_no_physical_motor_ID=True,counts_are_not_total_rotor_count=True,
            hidden_prior_in_observed_sum=False,hidden_force_owner='joint_unknown_state_d',
            anonymous_group_is_total_thrust=False,metric_thrust_measured=False,mechanical_RPM_measured=False,
            component_circumscribed_discs_disjoint=True,head_or_tail_required=False)
        return component,max(.35,float(structure.get('geometric_weight',0.))),info
