"""Causal short relative-thrust trend, independent of rotor numbering.

Only a full four-rotor sum or a current qualified pooled period is eligible.
An unidentified partial-rotor mean can change composition and is excluded.
This extends the already implemented headless-normal rollout; unknown force
is not multiplied by the blade-frequency trend. It is an experimental prior,
not measured future RPM or a calibrated propeller thrust curve.
"""
from collections import deque
import numpy as np

class ThrustTrendForecast:
    def __init__(self):self.rows=deque();self.last_t=None;self.source=None
    def observe(self,t,selection,pooled,amplitude_bridge=None):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Strictly increasing times required')
        self.last_t=t
        info=dict(qualified=False,reason='no_current_common_thrust_proxy',rotor_ids_used=False)
        source=selection.get('source');relative=selection.get('relative_f2')
        current=bool((amplitude_bridge or {}).get('learning_input_current',True))
        source_ok=source in ('four_current_squared_frequencies','four_slot_f2_sum',
            'pooled_flicker_representative_speed_hypothesis','pooled_axis_separate_gauge')
        if not current or not source_ok or not selection.get('coherent') or relative is None or relative<=0.:
            self.rows.clear();return info
        if source.startswith('pooled') and not pooled.get('qualified'):
            self.rows.clear();return info
        if source!=self.source or (self.rows and t-self.rows[-1][0]>.08+1e-8):self.rows.clear()
        self.source=source
        # Two log-frequency bins in f^2, inflated when the dominant mode has
        # weak spatial support. A proxy floor avoids fictitious precision from
        # many correlated rotor pixels or overlapping source windows.
        sigma=2*np.log(1.02)/max(.35,float(pooled.get('mode_fraction',1.) or .35))
        self.rows.append((t,np.log(relative),sigma))
        while self.rows and t-self.rows[0][0]>.35+1e-8:self.rows.popleft()
        if len(self.rows)<6 or t-self.rows[0][0]<.16-1e-8:return dict(info,reason='thrust_trend_warmup')
        a=np.asarray(self.rows);x=a[:,0]-t;w=1/a[:,2]**2;w/=w.sum()
        mx=w@x;my=w@a[:,1];slope=float(w@((x-mx)*(a[:,1]-my))/max(w@((x-mx)**2),1e-12))
        residual=a[:,1]-(my+slope*(x-mx));noise=max(float(np.sqrt(w@residual**2)),float(np.median(a[:,2])))
        se=noise/max(float(np.ptp(x)),.08);reliability=slope*slope/(slope*slope+se*se) if slope else 0.
        rate=float(np.clip(slope*reliability,-np.log(1.25)/.4,np.log(1.25)/.4))
        return dict(info,qualified=True,reason='current_same_source_relative_thrust_trend',
            log_thrust_rate_per_s=rate,raw_log_rate_per_s=slope,rate_standard_error_proxy=se,
            reliability=reliability,rows=len(a),span_s=float(np.ptp(x)),source=source,
            maximum_future_multiplier=1.25,minimum_future_multiplier=.8,
            unknown_force_scaled=False,uncertainty_is_uncalibrated=True)
    @staticmethod
    def correct(predictions,force,normal,trend,memory=None):
        result={k:list(v) for k,v in predictions.items()}
        info=dict(applied=False,reason=trend.get('reason'),trend=trend,unknown_force_scaled=False)
        if not predictions or not trend.get('qualified') or not force.get('active') or force.get('input_u') is None:return result,info
        rate=trend['log_thrust_rate_per_s'];omega=(normal or {}).get('trend',{}).get('shrunk_angular_rate_rad_s',0.) if (normal or {}).get('applied') else 0.
        tau=float(force.get('acceleration_correlation_s',.35));lam=-1/tau+1j*omega
        weight=float((memory or {}).get('weight',0.)) if (memory or {}).get('applied') else 0.
        from adaptive_force_memory import kernel
        known=float(force['effective_gain_px_s2'])*np.asarray(force['input_u']);z=complex(*known)
        changes={}
        for h,p in predictions.items():
            H=float(h)
            change=(1-weight)*(kernel(lam+rate,H)-kernel(lam,H))
            change+=weight*(kernel(1j*omega+rate,H)-kernel(1j*omega,H))
            d=change*z;delta=np.array([d.real,d.imag]);result[h]=(np.asarray(p)+delta).tolist();changes[h]=delta.tolist()
        return result,dict(info,applied=True,reason='known_relative_thrust_trend_rollout',corrections_px=changes)


class SourceAwareThrustTrendForecast(ThrustTrendForecast):
    """Keep separate actually observed histories of full sums and pooled modes.

    A selector choosing a partial-rotor mean need not erase an independently
    valid pooled observation from that same timestamp. Such pooled samples
    may train their own trend, but a pooled trend is NEVER applied to an
    unidentified partial-rotor mean. Only the matching current source is used.
    """
    full_sources=('four_current_squared_frequencies','four_slot_f2_sum')
    pooled_sources=('pooled_flicker_representative_speed_hypothesis','pooled_axis_separate_gauge')
    def __init__(self):
        super().__init__();self.histories={};self.epoch=None
    def _append(self,key,t,value,sigma):
        history=self.histories.setdefault(key,deque())
        if history and (t-history[-1][0]>.12+1e-8 or abs(value-history[-1][1])>=.16):history.clear()
        history.append((t,value,sigma))
        while history and t-history[0][0]>.35+1e-8:history.popleft()
    def observe(self,t,selection,pooled,amplitude_bridge=None,geometry_epoch=None):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Strictly increasing times required')
        self.last_t=t
        if self.epoch is not None and geometry_epoch is not None and geometry_epoch!=self.epoch:self.histories.clear()
        if geometry_epoch is not None:self.epoch=geometry_epoch
        for key in list(self.histories):
            history=self.histories[key]
            while history and t-history[0][0]>.35+1e-8:history.popleft()
            if not history:del self.histories[key]
        provenance=str(pooled.get('source') or 'native_camera_pixels')
        pool_key='pool:'+provenance
        fraction=max(.35,float(pooled.get('mode_fraction',1.) or .35))
        sigma=2*np.log(1.02)/fraction
        frequency=pooled.get('frequency_hz')
        pool_current=bool(pooled.get('qualified') and frequency is not None and np.isfinite(frequency) and 60.<=frequency<=1200.)
        if pool_current:
            # Temporal support may contain only four new sensor pixels. Its
            # old votes are not extra independent current-frequency samples.
            current_pixels=pooled.get('current_mode_pixels')
            if current_pixels is not None and current_pixels>0:sigma*=max(1.,np.sqrt(16./current_pixels))
            self._append(pool_key,t,2*np.log(float(frequency)),sigma)
        source=selection.get('source');relative=selection.get('relative_f2')
        current=bool((amplitude_bridge or {}).get('learning_input_current',True))
        ready=bool(current and selection.get('coherent') and relative is not None and np.isfinite(relative) and relative>0.)
        if ready and source in self.full_sources:self._append('full4',t,np.log(relative),2*np.log(1.02))
        info=dict(qualified=False,reason='no_matching_current_thrust_source',rotor_ids_used=False,
            current_frequency_invented=False,partial_rotor_mean_extrapolated=False,
            native_and_stabilized_histories_separated=True,source_histories={k:len(v) for k,v in self.histories.items()})
        if not ready:return dict(info,reason='current_amplitude_held_missing_or_incoherent')
        if source in self.full_sources:key='full4'
        elif source in self.pooled_sources and pool_current:key=pool_key
        else:return dict(info,reason='partial_rotor_composition_has_no_total_thrust_derivative')
        history=self.histories.get(key,deque())
        if len(history)<6 or t-history[0][0]<.16-1e-8:return dict(info,reason='source_specific_thrust_trend_warmup')
        a=np.asarray(history);x=a[:,0]-t;w=1/a[:,2]**2;w/=w.sum()
        if t-a[-1,0]>1e-8 or np.max(np.diff(a[:,0]))>.12+1e-8:
            return dict(info,reason='no_current_source_sample_or_long_gap')
        mx=w@x;my=w@a[:,1]
        slope=0. if np.ptp(a[:,1])==0. else float(w@((x-mx)*(a[:,1]-my))/max(w@((x-mx)**2),1e-12))
        residual=a[:,1]-(my+slope*(x-mx));noise=max(float(np.sqrt(w@residual**2)),float(np.median(a[:,2])))
        se=noise/max(float(np.ptp(x)),.08);reliability=slope*slope/(slope*slope+se*se) if slope else 0.
        rate=float(np.clip(slope*reliability,-np.log(1.25)/.4,np.log(1.25)/.4))
        return dict(info,qualified=True,reason='current_source_specific_thrust_trend',
            log_thrust_rate_per_s=rate,relative_thrust_rate_per_s=rate*float(relative),
            raw_log_rate_per_s=slope,rate_standard_error_proxy=se,reliability=reliability,
            rows=len(a),span_s=float(np.ptp(x)),source=source,history_source=key,
            maximum_future_multiplier=1.25,minimum_future_multiplier=.8,unknown_force_scaled=False,
            pooled_requires_representative_common_rotor_scale=source in self.pooled_sources,
            mechanical_RPM_measured=False,uncertainty_is_uncalibrated=True)
