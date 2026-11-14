"""Causal future net-force memory, retaining the same rotor/two-force state.

Three conditional rollouts of ONE state: existing0.35s, short0.1s, persistent.
Reuse the existing matured native-position robust simplex credit. No labels,
NN, recording identity, new position filter or future RPM is supplied.
"""
import copy
from collections import deque
import numpy as np
from adaptive_force_memory import AdaptiveForceMemory
from physical_publication import evaluate


def with_memory(row,tau):
    r=copy.deepcopy(row)
    def visit(force):
        force['acceleration_correlation_s']=tau
        for component in force.get('response_model_mixture',[]):visit(component['force'])
    visit(r['force']);return r


class ForcePersistenceRollout:
    def __init__(self):
        self.credit=AdaptiveForceMemory(allow_shorter=True);self.last_origin=None

    def step(self,row):
        t=row['t'];body=row['body'];motion=row.get('motion') or {};c=row['coefficient']
        shift=body.get('measurement_reference_shift_xy')
        if shift is not None:self.credit.rebase(shift)
        observed=bool(body.get('detected') and not body.get('center_provisional') and motion.get('current_observation'))
        z=body['center'] if observed else None
        _,credit=self.credit.step(t,c,z,{},row['force'],valid=row['valid'])
        if c is None:return copy.deepcopy(row)
        variants=[row,with_memory(row,.1),with_memory(row,1e12)]
        forecasts=[evaluate(r,[.1,.2,.4]) for r in variants]
        if row['valid'] and observed and (self.last_origin is None or t-self.last_origin>=.199):
            for h in (.1,.2,.4):
                base=np.asarray(forecasts[0][str(h)]);short=np.asarray(forecasts[1][str(h)])-base;long=np.asarray(forecasts[2][str(h)])-base
                if max(float(short@short),float(long@long))>=36.:
                    self.credit.pending.append(dict(target=t+h,base=base.copy(),delta=long.copy(),short_delta=short.copy(),horizon=h))
            self.credit.pending=deque(sorted(self.credit.pending,key=lambda r:r['target']));self.last_origin=t
        short=self.credit.short_weight;long=self.credit.weight;weights=[1-short-long,short,long]
        r=copy.deepcopy(row)
        r['force']['response_model_mixture']=[dict(weight=float(w),coefficient=v['coefficient'],force=v['force'],normal_turn_forecast=v.get('normal_turn_forecast')) for w,v in zip(weights,variants) if w>0.]
        r['force']['future_net_force_memory_is_model_mixture']=True
        r['state']['future_force_persistence']=dict(weights=dict(original=float(weights[0]),short=float(short),persistent=float(long)),
            matured_rows=len(self.credit.rows),pending_targets=len(self.credit.pending),updated=credit['updated'],
            calibration_observation='own_current_filtered_native_rotor_position',
            no_current_frequency_required_for_unknown_force_memory=True,labels_used=False,
            prediction_is_not_future_command_measurement=True,same_state_and_observed_unknown_partition=True,
            geometry_rotors_and_current_state_unchanged=True,recording_identity_used=False,
            uncertainty_not_calibrated=True)
        r['physical']=evaluate(r,[.1,.2,.4]);return r
