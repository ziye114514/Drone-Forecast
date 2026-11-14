"""Retain physical learning in a fixed frequency-reference coordinate system."""
from pathlib import Path
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage28'),str(ROOT/'research/stage48')]
from axis28 import Predictor,prior
from compose48 import associate,observation,empty_sensing

METHODS=['reset_all','preserve_axis','stable_reference']
PARAMS=dict(reset_gap_s=prior.robust.PARAMS['reset_gap_s'],canonical_ratio_bounds=prior.PARAMS['ratio_clip'],
    reference='Keep first qualified aggregate S0 in a fixed model coordinate. Feed s=S/S0 even if sensor local denominator changes. Equivalent to k_new=k_old*S0_new/S0_old, P_new=D P_old D^T, queued_s_new=queued_s_old*S0_old/S0_new, but canonical coordinates avoid changing numeric bounds and excitation reference.',
    isolation='Original shared frequency/axis quality gate, four-visible learning gate,0.6s warmup, loss and RLS/rollout arithmetic unchanged. No attempt to infer true mass/wind. NN500ms object gap still resets all state. Reference ratio outside old0.2..5 domain is rejected rather than silently clipped.',
    identity='Association uses current body box gate from48. Aggregate sum-square is permutation invariant, but can still be wrong under false rotor/harmonic selection; continuity of object identity is assumed only across original short body gaps.')

class Reference:
    def __init__(self):self.total=None
    def clear(self):self.total=None
    def transform(self,sensed):
        result=copy.deepcopy(sensed);valid=bool(sensed.get('four_frequencies') and sensed.get('thrust_ratio') is not None and sensed.get('sum_frequency_squared') is not None)
        d=dict(reference_total=self.total,qualified=valid,range_rejected=False)
        if not valid:return result,d
        S=float(sensed['sum_frequency_squared']);s=float(sensed['thrust_ratio'])
        if not np.isfinite(S) or not np.isfinite(s) or S<=0 or s<=0:
            result.update(four_frequencies=False,thrust_ratio=None);d.update(qualified=False,reason='nonpositive_or_nonfinite');return result,d
        local_total=S/s
        if self.total is None:self.total=local_total
        canonical=S/self.total
        d.update(reference_total=self.total,local_reference_total=local_total,local_ratio=s,canonical_ratio=canonical)
        if not PARAMS['canonical_ratio_bounds'][0]<=canonical<=PARAMS['canonical_ratio_bounds'][1]:
            result.update(four_frequencies=False,thrust_ratio=None);d.update(range_rejected=True)
        else:result['thrust_ratio']=canonical
        return result,d

class Managed:
    def __init__(self,method):
        assert method in METHODS;self.method=method;self.predictor=Predictor('canonical_static');self.reference=Reference();self.epoch=0
    def step(self,t,obs,sensed,epoch):
        p=self.predictor;gap=bool(p.baseline.base.last_seen is not None and t-p.baseline.base.last_seen>PARAMS['reset_gap_s'])
        if gap:self.reference.clear()
        changed=epoch!=self.epoch;self.epoch=epoch
        if changed:
            if self.method=='reset_all':p.reset_features()
            elif self.method=='preserve_axis':
                axis=p.axis;model=p.models['adaptive_axis_only'];p.reset_features();p.axis=axis;p.models['adaptive_axis_only']=model
            else:
                # Previous force freshness belongs to the old sensor epoch;
                # wait for a qualified new aggregate before enabling it again.
                p.last_force_t=None
        if self.method=='stable_reference':
            sensed,d=self.reference.transform(sensed)
            if d['range_rejected']:p.last_force_t=None
        else:d=dict(reference_total=None,range_rejected=False)
        before={name:model.updates for name,model in p.models.items()}
        q,_=p.step(dict(t=t,observation=obs,sensing=sensed))
        delta={name:max(0,model.updates-before[name]) for name,model in p.models.items()}
        return q,dict(epoch=epoch,epoch_changed=changed,object_gap=gap,reference=d,actual_update_increments=delta)
