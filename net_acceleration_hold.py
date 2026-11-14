"""Short inertial net-acceleration prior after a rejected motion derivative.

A current measured velocity and plane axis must still support the previous
qualified acceleration. The original measurement timestamp is never refreshed
by holding. No period, rotor or new acceleration measurement is invented.
"""
import numpy as np
from axis_amplitude_bridge import current_axis_qualified

class NetAccelerationHold:
    def __init__(self):self.last=None;self.last_t=None
    def apply(self,t,coefficient,predictions,force,motion,structure):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing cutoffs required')
        self.last_t=t;info=dict(applied=False,reason='no_recent_qualified_net_acceleration',maximum_age_s=.12,
            current_acceleration_measured=False,current_period_invented=False,refreshes_measurement_time=False)
        m=motion or {};a=m.get('acceleration') or {};obs=m.get('observation') or {}
        if a.get('applied') and obs.get('valid'):
            self.last=(t,np.asarray(a['shrunk_acceleration']),np.asarray(a['standard_error_proxy']),dict(obs))
            return predictions,force,dict(info,reason='current_qualified_acceleration_preserved')
        if self.last is None or coefficient is None or not obs.get('valid') or not current_axis_qualified(structure):return predictions,force,info
        if force.get('active') or force.get('short_gap_acceleration_held'):return predictions,force,dict(info,reason='existing_physical_input_or_hold_preserved')
        stamp,acceleration,se,old=self.last;age=t-stamp
        if age>.12+1e-9:return predictions,force,dict(info,reason='net_acceleration_prior_expired',age_s=age)
        velocity=np.asarray(obs['velocity']);old_v=np.asarray(old['velocity'])
        if np.linalg.norm(velocity)<60.:return predictions,force,dict(info,reason='stationary_or_very_slow_translation_protected')
        delta=obs['midpoint_t']-old['midpoint_t']
        if delta<=0. or obs['midpoint_t']>t:return predictions,force,dict(info,reason='invalid_velocity_times')
        expected=old_v+delta*acceleration;error=velocity-expected
        # These quantities share images and overlapping fits. Use a worst-case
        # covariance-sum bound, not a fictitious independent fused measurement.
        covariance=3*(np.asarray(obs['velocity_covariance'])+np.asarray(old['velocity_covariance'])+np.diag((delta*se)**2))
        chi2=float(error@np.linalg.solve(covariance,error))
        if not np.isfinite(chi2) or chi2>5.991464547107979:
            return predictions,force,dict(info,reason='current_velocity_rejects_inertial_acceleration',chi2_proxy=chi2)
        tau=float(force.get('acceleration_correlation_s',.35));held=acceleration*np.exp(-age/tau)
        p,v=np.asarray(coefficient['position']),np.asarray(coefficient['velocity'])
        out={str(h):(p+h*v+tau*(h+tau*np.expm1(-h/tau))*held).tolist() for h in (.1,.2,.4)}
        f=dict(force,acceleration_px_s2=held.tolist(),acceleration_source='short_gap_measured_net_acceleration_prior')
        return out,f,dict(info,applied=True,reason='current_velocity_supports_short_net_acceleration_prior',
            original_measurement_t=stamp,age_s=age,chi2_proxy=chi2,uncertainty_is_proxy=True,
            held_acceleration_px_s2=held.tolist(),gain_and_disturbance_not_updated=True)
