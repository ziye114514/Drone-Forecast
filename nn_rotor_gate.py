"""Causal, optional native-rotor residual on an unchanged NN forecast.

Reuse AdaptiveForceMemory's matured-own-position/Huber/ridge mechanism for
an external candidate. No labels, second position filter, or offline fit.
The initial weight is zero; past candidate success earns its current weight.
"""
from collections import deque
import copy
import numpy as np
from physical_publication import evaluate


def evaluate_gated(row,horizons):
    result=evaluate(row,horizons)
    gate=row.get('rotor_residual_fusion') or {}
    if not gate.get('applied'):return result
    other=evaluate(gate['candidate'],horizons);w=float(gate['weight'])
    return {h:(np.asarray(p)+w*(np.asarray(other[h])-p)).tolist() for h,p in result.items()}


class CausalRotorResidualGate:
    def __init__(self):
        self.pending=deque();self.rows=deque();self.previous=None
        self.last_origin=None;self.last_t=None;self.weight=0.

    def step(self,base,candidate,current_evidence):
        t=float(base['t']);motion=base.get('image_motion') or {}
        observation=motion.get('observation') or {}
        reanchor=bool(observation.get('anchor_upgrade') or observation.get('visibility_reanchor')
            or observation.get('contained_weak_patch_motion'))
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing cutoffs required')
        reset=bool(reanchor or self.last_t is not None and t-self.last_t>.5)
        if reset:self.__init__()
        self.last_t=t;updated=False
        z=base.get('force_observed_position') if base['valid'] else None
        if z is not None:
            z=np.asarray(z,float)
            if self.previous is not None:
                prev,old=self.previous
                while self.pending and self.pending[0]['target']<=t+1e-8:
                    r=self.pending.popleft();target=r['target']
                    if not prev-1e-8<=target<=t+1e-8 or t-prev>.08+1e-8:continue
                    fraction=0. if t==prev else (target-prev)/(t-prev)
                    target_position=(1-fraction)*old+fraction*z
                    self.rows.append((t,r['delta'],target_position-r['base']));updated=True
            self.previous=(t,z.copy())
        while self.rows and t-self.rows[0][0]>3.:self.rows.popleft()
        if updated:
            numerator=0.;denominator=36.
            for stamp,delta,error in self.rows:
                residual=error-self.weight*delta
                robust=min(1.,6./max(float(np.linalg.norm(residual)),1e-9))
                w=np.exp(-(t-stamp)/.35)*robust
                numerator+=w*float(delta@error);denominator+=w*float(delta@delta)
            self.weight=float(np.clip(numerator/denominator,0.,1.))
        eligible=bool(current_evidence and base['valid'] and base.get('coefficient') is not None)
        if eligible and z is not None and (self.last_origin is None or t-self.last_origin>=.199):
            p=np.asarray(evaluate(base,[.4])['0.4'])
            delta=np.asarray(evaluate(candidate,[.4])['0.4'])-p
            if float(delta@delta)>=36.:
                self.pending.append(dict(target=t+.4,base=p,delta=delta))
            self.last_origin=t
        row=copy.deepcopy(base)
        row['rotor_residual_fusion']=dict(weight=self.weight,applied=bool(eligible and self.weight>0.),
            current_evidence=bool(current_evidence),matured_rows=len(self.rows),updated=updated,
            pending_targets=len(self.pending),visibility_reset=reset,
            candidate=copy.deepcopy(candidate),calibration_target='matured_own_NN_positions_at_0.4s',
            annotations_used=False,future_observations_used_before_maturity=False,
            NN_position_velocity_validity_preserved=True,metric_force_identified=False)
        row['physical']=evaluate_gated(row,[.1,.2,.4]) if base.get('coefficient') is not None else {}
        return row
