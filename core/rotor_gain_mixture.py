"""Causal positive mechanical-gain uncertainty, not selection on FDE labels.

Seven log-gain quadrature members span the existing25x gain bound and its
reciprocal. They all see identical current/past planes and rotor samples.
Only later plane innovations change their weights. Output is their expected
future rollout; no recording chooses a different hyperparameter.
"""
import copy
import numpy as np
from rotor_torque_innovation import RotorTorqueInnovation,adapt


class RotorGainMixture:
    def __init__(self):
        self.scales=np.geomspace(1/25,25,7);self.models=[RotorTorqueInnovation(gain_scale=g,fixed_gain=True,frequency_changes_only=True) for g in self.scales]
        weight=np.ones(7);weight[[0,-1]]=.5;weight/=weight.sum();self.log_weight=np.log(weight);self.last_weight_t=None

    def step(self,row,period):
        infos=[m.observe(row,period) for m in self.models];t=row['t']
        likelihood=np.asarray([i.get('past_plane_log_likelihood_ratio_vs_matched_noise_control',np.nan) for i in infos]);matured=bool(np.isfinite(likelihood).all())
        if matured and (self.last_weight_t is None or t-self.last_weight_t>=.020-1e-8):
            self.log_weight+=likelihood;self.log_weight-=np.max(self.log_weight);self.last_weight_t=t
        weight=np.exp(self.log_weight-np.max(self.log_weight));weight/=weight.sum()
        result=copy.deepcopy(row);minimal=dict(coefficient=row['coefficient'],force=row['force'],normal_turn_forecast=row.get('normal_turn_forecast'))
        members=[]
        for w,info in zip(weight,infos):
            r=adapt(minimal,info);r['weight']=float(w);members.append(r)
        result['force']['response_model_mixture']=members
        result['rotor_mechanical_gain_mixture']=dict(scales=self.scales.tolist(),weights=weight.tolist(),
            posterior_mean_scale=float(weight@self.scales),posterior_entropy=float(-np.sum(weight*np.log(np.maximum(weight,1e-300)))),
            current_scale_selection_on_FDE=False,updates_use_only_matured_past_plane_responses=True,
            current_plane_likelihood_used=matured,absolute_metric_thrust_or_inertia_measured=False,
            evidence_is_likelihood_ratio_to_matched_input_noise_control=True,
            geometry_change_counted_as_thrust_change=False,
            common_input_drive=next((i.get('drive_change') for i in infos if 'drive_change' in i),0.),
            member_angular_corrections=[dict(applied=i.get('applied',False),rate=i.get('delta_rate_rad_s',0.),
                acceleration=i.get('delta_acceleration_rad_s2',0.)) for i in infos])
        return result
