"""Past/current visible f_i^2 change drives future observed-force change.

Sum CURRENT SHARED sources, without assuming equal rotor forces or imputing
hidden frequencies. Geometry motion alone cannot create this derivative.
The inherited4% weight-noise proxy shrinks an unresolved change toward zero;
it is not calibrated RPM accuracy. A reinitialized phase is a new source.
"""
import copy
import numpy as np
from rotor_frequency_drive import sources
from axis_amplitude_bridge import current_axis_qualified


class VisibleForceRate:
    def __init__(self):self.previous={};self.t=None;self.epoch=None

    def step(self,row,period=None):
        period=row.get('period_selection') or {} if period is None else period
        current=sources(row,period);t=row['t'];dt=None if self.t is None else t-self.t
        epoch=row['structure'].get('geometry_epoch');same=self.epoch==epoch
        shared=sorted(k for k in current.keys()&self.previous.keys() if current[k]['instance']==self.previous[k]['instance']) if same and dt and 0<dt<=.040001 else []
        n=np.asarray(row['structure'].get('axis_xy') or [],float)
        result=copy.deepcopy(row);info=dict(applied=False,shared_current_sources=shared,
            source_count=len(current),source_membership_is_not_thrust_change=True,
            hidden_force_stays_unknown=True,mechanical_RPM_or_Newton_thrust_measured=False)
        if shared and row['coefficient'] and current_axis_qualified(row['structure']) and n.shape==(2,) and np.linalg.norm(n)>1e-6:
            change=sum(current[k]['weight']-self.previous[k]['weight'] for k in shared)/dt
            variance=sum(.04**2*(current[k]['weight']**2+self.previous[k]['weight']**2) for k in shared)/dt**2
            reliability=change*change/(change*change+max(variance,1e-12))
            length=float(np.linalg.norm(n));direction=n/length
            if direction[1]>0:direction=-direction
            def apply(packet):
                f=packet['force'];normal=packet.get('normal_turn_forecast') or {}
                raw=float(f['effective_gain_px_s2'])*length*change
                rate=raw*reliability
                if not abs(rate)>1e-12:return
                f['visible_force_rate']=dict(amplitude_rate_px_s3=rate,raw_amplitude_rate_px_s3=raw,
                    variance_proxy_px2_s6=(float(f['effective_gain_px_s2'])*length)**2*variance,
                    reliability=reliability,correlation_s=.35,normal_xy=direction.tolist(),
                    per_source_unequal_f2_used=True,hidden_force_not_in_measured_derivative=True)
                for part in f.get('response_model_mixture',[]):apply(part)
            apply(result);info.update(applied=bool(abs(change)>1e-12),f2_rate=change,variance_proxy=variance,reliability=reliability)
        self.previous=current;self.t=float(t);self.epoch=epoch
        result['visible_force_rate']=info
        return result
