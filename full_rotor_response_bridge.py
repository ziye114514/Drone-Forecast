"""Full-current-rotor normalization of an ESTIMATED image normal response.

When all four current periods are independently supported, q=sum(f_i^2)/ref
is complete. The existing image-normal response A supplies an effective A/q
prior; it is not mechanical CT/m, true thrust, or an independent sensor.
Missing periods never get filled. The learned intrinsic angular response may
remain a parameter prior during partial visibility while the target is tracked.
"""
import copy
import numpy as np
from rotor_torque_innovation import RotorTorqueInnovation,adapt
from axis_amplitude_bridge import current_axis_qualified


def normal_response(row,n):
    parts=[]
    def collect(r,weight):
        f=r['force'];mixture=f.get('response_model_mixture')
        if mixture:
            for part in mixture:collect(part,weight*part['weight'])
            return
        u=f.get('input_u');known=0. if u is None else f['effective_gain_px_s2']*float(n@np.asarray(u)/(n@n))
        h=float(f.get('unobserved_normal_response_gain',0.));sigma=float(f.get('unobserved_normal_response_sigma',1000.))
        parts.append((weight,h+known,sigma*sigma))
    collect(row,1.);mean=sum(w*x for w,x,_ in parts);variance=sum(w*(v+(x-mean)**2) for w,x,v in parts)
    return float(mean),float(np.sqrt(max(variance,0.)))


class FullRotorResponseBridge:
    def __init__(self,*,redistribution_only=False):
        self.redistribution_only=bool(redistribution_only)
        self.model=RotorTorqueInnovation(frequency_changes_only=True,redistribution_only=self.redistribution_only);self.calibration=None;self.epoch=None;self.last_body_t=None

    def step(self,row,period):
        t=row['t'];epoch=row['structure'].get('geometry_epoch');current=[];updated=False
        if self.epoch!=epoch or self.last_body_t is not None and t-self.last_body_t>.5:
            self.calibration=None;self.model=RotorTorqueInnovation(frequency_changes_only=True,redistribution_only=self.redistribution_only)
        self.epoch=epoch
        if row['body'].get('detected'):self.last_body_t=t
        n=np.asarray(row['structure'].get('axis_xy') if row['structure'].get('axis_xy') is not None else [],float)
        for r in period.get('meta',{}).get('records',[]):
            if r.get('qualified') and r.get('current_force_admitted'):current.append(r)
        complete=len(current)==4 and len({r.get('model_slot') for r in current})==4
        if complete and row['valid'] and current_axis_qualified(row['structure']) and n.shape==(2,) and n@n>1e-8:
            q=sum(r['proxy_twofold_hz']**2/360000 for r in current)
            amplitude,sigma=normal_response(row,n);prior=self.model.initial_gain(row,period)
            if amplitude>0 and q>0 and prior is not None:
                trust=amplitude*amplitude/max(amplitude*amplitude+sigma*sigma,1e-12)
                original=float(row['force']['effective_gain_px_s2']);estimate=amplitude/q
                # Model-conditioned normalization, with a broad covariance.
                # No independent-measurement likelihood is applied to data
                # already used by the image-position observer.
                effective=(1-trust)*original+trust*estimate
                gamma=prior['gain']*effective/original
                sd=abs(gamma)*np.sqrt(1+(sigma/max(amplitude,1e-12))**2)
                self.calibration=dict(t=t,reference='all_four_current_validated_periods',visible_count=4,
                    normal_response_estimate=amplitude,normal_response_sigma_proxy=sigma,
                    measured_relative_f2_sum=float(q),normalization_reliability=float(trust),
                    default_image_gain=original,image_gain_from_normal_response=estimate,
                    effective_image_gain=float(effective),angular_gain_prior=float(gamma),angular_gain_sigma_proxy=float(sd),
                    independent_thrust_or_inertia_calibration=False)
                updated=True
                if self.model.x is None:self.model.gain_scale=effective/original
                else:
                    self.model.x[3]=gamma;self.model.P[3,:3]=0.;self.model.P[:3,3]=0.
                    self.model.P[3,3]=max(float(self.model.P[3,3]),sd*sd)
        info=self.model.observe(row,period);result=adapt(row,info)
        result['full_rotor_response_bridge']=dict(calibration_updated=updated,calibration=copy.deepcopy(self.calibration),
            current_complete_periods=complete,hidden_frequencies_added=0,
            current_net_angular_acceleration_not_reset_by_parameter_update=True,
            translation_gain_or_unknown_force_reassigned=False,normal_response_is_not_independent_true_thrust=True)
        return result
