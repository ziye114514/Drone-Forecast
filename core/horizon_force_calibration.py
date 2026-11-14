"""Horizon-specific calibration of the existing one-state physical rollout.

Reuses CalibratedResponsiveForce's matured-position response calibration, with
separate horizons and the existing Huber/noise conventions. This estimates
future NET-force contribution reliability, not actual thrust, mass or RPM.
No second trajectory filter, new image flow, or annotations are introduced.
"""
from collections import deque
import numpy as np


class HorizonForceCalibration:
    def __init__(self):
        self.pending=deque();self.rows={};self.weights={};self.previous=None;self.last_t=None;self.last_origin=None

    def step(self,t,c,observed_position,predictions,valid=True):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing cutoffs required')
        if self.previous is not None and t-self.previous[0]>.5:self.__init__()
        self.last_t=t;updated=[];reference=None
        if observed_position is not None:
            z=np.asarray(observed_position,float);reference=z
            if self.previous is not None:
                old_t,old=self.previous
                while self.pending and self.pending[0]['target']<=t+1e-8:
                    r=self.pending.popleft();stamp=r['target']
                    if not old_t-1e-8<=stamp<=t+1e-8 or t-old_t>.08+1e-8:continue
                    mix=0. if old_t==t else float(np.clip((stamp-old_t)/(t-old_t),0.,1.))
                    actual=(1-mix)*old+mix*z;key=r['key']
                    self.rows.setdefault(key,deque()).append((t,r['delta'],actual-r['base']))
                    if key not in updated:updated.append(key)
            self.previous=(t,z.copy())
        for key,rows in self.rows.items():
            while rows and t-rows[0][0]>3.:rows.popleft()
        for key in updated:
            old=self.weights.get(key,1.);numerator=36.;denominator=36.
            for stamp,delta,error in self.rows[key]:
                residual=error-old*delta
                w=np.exp(-(t-stamp)/.35)*min(1.,6./max(float(np.linalg.norm(residual)),1e-9))
                numerator+=w*float(delta@error);denominator+=w*float(delta@delta)
            self.weights[key]=float(np.clip(numerator/denominator,0.,1.))
        result={k:list(v) for k,v in predictions.items()};changes={}
        if c is not None and predictions:
            p=np.asarray(c['position']);v=np.asarray(c['velocity'])
            origin=bool(valid and reference is not None and (self.last_origin is None or t-self.last_origin>=.199))
            for key,prediction in predictions.items():
                h=float(key);base=p+h*v;delta=np.asarray(prediction)-base
                weight=self.weights.get(key,1.);value=base+weight*delta
                result[key]=value.tolist();changes[key]=(value-np.asarray(prediction)).tolist()
                if origin and float(delta@delta)>=36.:
                    self.pending.append(dict(target=t+h,key=key,base=base.copy(),delta=delta.copy()))
            if origin:self.last_origin=t;self.pending=deque(sorted(self.pending,key=lambda r:r['target']))
        return result,dict(applied=bool(any(abs(self.weights.get(k,1.)-1.)>1e-12 for k in changes)),
            response_reliability_by_horizon=dict(self.weights),updated_horizons=updated,corrections_px=changes,
            matured_rows={k:len(v) for k,v in self.rows.items()},pending_targets=len(self.pending),
            prior_response_reliability=1.,current_force_not_reidentified=True,
            current_position_velocity_and_validity_unchanged=True,second_trajectory_filters=0,
            annotations_used=False,future_observations_used_before_maturity=False,
            target='future net-force contribution, not measured rotor thrust')
