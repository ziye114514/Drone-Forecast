"""Causal credit for the verified directional-force increment over191.

Reuse AdaptiveForceMemory's matured observations, reference rebasing and
robust scalar update. Only the queued delta changes: separated minus191.
No coordinate labels, parameter search, NN or current future target is used.
"""
import copy
import numpy as np
from adaptive_force_memory import AdaptiveForceMemory
from physical_publication import evaluate
from axis_amplitude_bridge import current_axis_qualified

class CausalResponseCredit:
    def __init__(self):self.memory=AdaptiveForceMemory();self.last_origin=None

    def step(self,baseline,alternative):
        t=baseline['t'];body=baseline['body'];motion=baseline.get('motion') or {};c=baseline['coefficient']
        shift=body.get('measurement_reference_shift_xy')
        if shift is not None:self.memory.rebase(shift)
        observed=body.get('detected') and not body.get('center_provisional',False) and motion.get('current_observation',False)
        z=body['center'] if observed else None
        self.memory.step(t,c,z,{},baseline['force'],valid=baseline['valid'])
        eligible=bool(c is not None and alternative['coefficient'] is not None and baseline['valid'] and alternative['valid'] and current_axis_qualified(baseline['structure']))
        weight=self.memory.weight if eligible else 0.
        if eligible and observed and (self.last_origin is None or t-self.last_origin>=.199):
            aa=evaluate(baseline,[.1,.2,.4]);bb=evaluate(alternative,[.1,.2,.4])
            for h in (.1,.2,.4):
                a=np.asarray(aa[str(h)]);delta=np.asarray(bb[str(h)])-a
                if float(delta@delta)>=36.:
                    self.memory.pending.append(dict(target=t+h,base=a.copy(),delta=delta.copy(),horizon=h))
            from collections import deque
            self.memory.pending=deque(sorted(self.memory.pending,key=lambda r:r['target']));self.last_origin=t
        result=copy.deepcopy(baseline)
        if c is not None and alternative['coefficient'] is not None:
            for name in ('position','velocity','acceleration'):
                result['coefficient'][name]=((1-weight)*np.asarray(c[name])+weight*np.asarray(alternative['coefficient'][name])).tolist()
            f=result['force'];f['acceleration_px_s2']=result['coefficient']['acceleration']
            f['effective_gain_px_s2']=(1-weight)*baseline['force']['effective_gain_px_s2']+weight*alternative['force']['effective_gain_px_s2']
            if baseline['force'].get('disturbance_px_s2') is not None and alternative['force'].get('disturbance_px_s2') is not None:
                f['disturbance_px_s2']=((1-weight)*np.asarray(baseline['force']['disturbance_px_s2'])+weight*np.asarray(alternative['force']['disturbance_px_s2'])).tolist()
            for field in ('unobserved_normal_response_px_s2','external_unknown_not_rotated','unobserved_normal_response_gain'):
                if baseline['force'].get(field) is not None and alternative['force'].get(field) is not None:
                    mixed=(1-weight)*np.asarray(baseline['force'][field])+weight*np.asarray(alternative['force'][field])
                    f[field]=mixed.tolist()
            f['force_parts_are_response_mixture_means']=True
            f['response_model_mixture']=[dict(weight=float(w),coefficient=r['coefficient'],force=r['force'],normal_turn_forecast=r.get('normal_turn_forecast')) for w,r in [(1-weight,baseline),(weight,alternative)]]
            for field,coefficient_field in [('position_sigma_px','position'),('velocity_sigma_px_s','velocity')]:
                if field in baseline['state'] and field in alternative['state']:
                    variance=(1-weight)*np.square(baseline['state'][field])+weight*np.square(alternative['state'][field])
                    variance+=weight*(1-weight)*np.square(np.asarray(c[coefficient_field])-np.asarray(alternative['coefficient'][coefficient_field]))
                    result['state'][field]=np.sqrt(variance).tolist()
        result['state']['causal_response_credit']=dict(weight=float(weight),fitted_weight=float(self.memory.weight),
            mature_rows=len(self.memory.rows),pending_targets=len(self.memory.pending),
            current_native_position_observed=bool(observed),current_plane_qualified=current_axis_qualified(baseline['structure']),
            annotations_used=False,future_observations_used_before_maturity=False,starts_with_original191=True,
            shared_observations_not_independent_tracks=True)
        result['physical']=evaluate(result,[.1,.2,.4]) if c else {}
        return result
