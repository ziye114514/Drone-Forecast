"""Observed-rotor dynamics with a gauge-invariant future input increment.

NN may provide frozen localization/velocity observations. No NN forecast is
blended. Hidden thrust belongs to d. Partial sources are not a full motor bank.
Reuses existing local-flow gain identification and headless normal fitting.
"""
from collections import deque,Counter
import copy
import numpy as np
from rotor_thrust_bank import RotorThrustBankProxy
from normal_turn_forecast import NormalTurnForecast
from flow_identifiable_gain import fit_gain
from adaptive_force_memory import kernel
from axis_amplitude_bridge import current_axis_qualified


def scale_aware_gain(velocities,inputs):
    available=np.array([u for _,u in inputs if u is not None],float)
    excitation=float(np.sqrt(np.mean(np.sum((available-available.mean(0))**2,axis=1)))) if len(available) else 0.
    # The arbitrary unit convention f²/(4*300²) must not decide whether an
    # identifiable gain is accepted. Bound its observed acceleration response,
    # reusing FastForceResponse's existing3000px/s²/excitation limit.
    limit=3000./max(excitation,1e-12)
    candidate,info=fit_gain(velocities,inputs,maximum_gain=limit)
    return candidate,dict(info,effective_gain_upper_bound=limit,input_excitation_rms=excitation,
        bound_basis='observed_response3000px_s2_not_absolute_coefficient5000',input_unit_invariant=True)


def absolute_components(meta,pooled,structure):
    """Positive amplitudes on current image normal; not the subtracted gauge."""
    n=np.asarray(structure.get('axis_xy') if structure.get('axis_xy') is not None else [],float)
    if n.shape!=(2,) or not np.isfinite(n).all() or np.linalg.norm(n)<1e-8:return {},None
    # Retain projection magnitude; only the sign is the existing image-up prior.
    if n[1]>0:n=-n
    length=float(np.linalg.norm(n));unit=n/length
    if meta.get('rotor_bank_used'):
        return {k:float(np.linalg.norm(v)) for k,v in meta.get('observed_components',{}).items()},unit
    f=pooled.get('frequency_hz')
    if not meta.get('coherent') or not pooled.get('qualified') or f is None:return {},unit
    key='pool:'+str(pooled.get('selection_source') or meta.get('pooled_estimator_source'))
    return {key:float(f*f/360000.*length)},unit


class VisibleForceDynamics:
    def __init__(self,learn=False,gain_window_s=.35):
        self.proxy=RotorThrustBankProxy(vector_pool=True);self.normal=NormalTurnForecast()
        self.histories={};self.inputs=deque();self.velocities=deque()
        self.signature=None;self.previous_t=None;self.epoch=None
        self.learn=learn;self.gain=200.;self.gain_sigma=None;self.updated_total=0
        self.gain_window_s=float(gain_window_s)
        self.gain_source=None;self.source_gains={}
        self.reasons=Counter()

    def step(self,base,observation,motion):
        t=float(base['t']);s=observation['structure'];pooled=observation.get('pooled') or {}
        raw,q,meta=self.proxy.step(t,s,observation['geometry'].get('rotors',[]),observation.get('rates',[]),pooled)
        current=bool(observation.get('current_pose') and current_axis_qualified(s) and raw is not None)
        components,n=absolute_components(meta,pooled,s) if current else ({},None)
        source='identified_visible_sum' if meta.get('rotor_bank_used') else next(iter(components),None)
        if source is not None and source!=self.gain_source:
            if self.gain_source is not None:self.source_gains[self.gain_source]=(self.gain,self.gain_sigma)
            self.gain,self.gain_sigma=self.source_gains.get(source,(200.,None));self.gain_source=source
        signature=tuple(sorted(components))
        flow=(motion or {}).get('observation') or {};epoch=flow.get('history_segment_epoch')
        reference_change=bool(flow.get('anchor_upgrade') or flow.get('visibility_reanchor') or flow.get('contained_weak_patch_motion'))
        changed=signature!=self.signature or epoch is not None and self.epoch is not None and epoch!=self.epoch
        if changed or reference_change or self.previous_t is not None and t-self.previous_t>.12+1e-8:
            self.inputs.clear();self.velocities.clear()
        self.signature=signature;self.previous_t=t;self.epoch=epoch
        for key in list(self.histories):
            h=self.histories[key]
            while h and t-h[0][0]>.35+1e-8:h.popleft()
            if not h:del self.histories[key]
        rates={};rate_info={}
        for key,value in components.items():
            h=self.histories.setdefault(key,deque())
            if h and (t-h[-1][0]>.12+1e-8 or abs(np.log(value)-h[-1][1])>=.16):h.clear()
            h.append((t,np.log(max(value,1e-12))))
            info=dict(qualified=False,reason='per_visible_source_trend_warmup',rows=len(h))
            rate=0.
            if len(h)>=6 and t-h[0][0]>=.16-1e-8:
                a=np.array(h);x=a[:,0]-t;y=a[:,1];x=x-x.mean()
                slope=float(x@(y-y.mean())/max(x@x,1e-12))
                noise=max(float(np.std(y-(y.mean()+slope*x))),2*np.log(1.02))
                se=noise/max(float(np.ptp(x)),.08)
                rate=float(np.clip(slope*slope/(slope*slope+se*se)*slope if slope else 0.,-np.log(1.25)/.4,np.log(1.25)/.4))
                info.update(qualified=True,reason='current_visible_source_trend',rate_per_s=rate,standard_error_proxy=se)
            rates[key]=rate;rate_info[key]=info
        # Keep timestamps, not a past mean acceleration compared to today's u.
        if components:self.inputs.append((t,sum(components.values())*n))
        else:self.inputs.append((t,None))
        if flow.get('valid') and flow.get('derivative_eligible',True) and components:
            R=np.asarray(flow['velocity_covariance'],float)
            self.velocities.append((float(flow['midpoint_t']),np.asarray(flow['velocity'],float),R,np.linalg.inv(np.linalg.cholesky(R))))
        while self.inputs and t-self.inputs[0][0]>self.gain_window_s+.15+1e-8:self.inputs.popleft()
        while self.velocities and t-self.velocities[0][0]>self.gain_window_s+1e-8:self.velocities.popleft()
        # First velocity midpoint precedes the first current input after a
        # source restart. Do not let that single uncovered row veto every
        # later fully covered observation until the whole window expires.
        covered=[r for r in self.velocities if self.inputs and self.inputs[0][0]-1e-8<=r[0]<=self.inputs[-1][0]+1e-8]
        candidate,fit=(scale_aware_gain(covered,self.inputs) if self.learn else fit_gain(covered,self.inputs)) if components else (None,dict(reason='no_current_observed_rotor_input'))
        fit=dict(fit,uncovered_midpoints_excluded=len(self.velocities)-len(covered))
        updated=False
        if self.learn and candidate is not None:
            sigma=fit['standard_error_proxy'];alpha=candidate*candidate/(candidate*candidate+sigma*sigma)
            self.gain+=alpha*(candidate-self.gain);self.gain_sigma=sigma;updated=True;self.updated_total+=1
        self.reasons[fit['reason']]+=1
        c=base.get('coefficient');trend=self.normal.observe(t,s,c,motion)
        omega=float(trend.get('shrunk_angular_rate_rad_s',0.)) if trend.get('qualified') else 0.
        a=np.asarray(c['acceleration'],float) if c else np.zeros(2)
        u=sum(components.values())*n if components else np.zeros(2)
        d=a-self.gain*u
        info=dict(current=bool(components),components=components,normal=None if n is None else n.tolist(),
            log_rates_per_s=rates,rate_details=rate_info,normal_trend=trend,omega=omega,
            effective_gain=self.gain,gain_sigma_proxy=self.gain_sigma,gain_updated=updated,gain_fit=fit,
            gain_history_window_s=self.gain_window_s,
            gain_source=self.gain_source,source_specific_gain=True,
            unknown_current=d.tolist(),current_net_acceleration=a.tolist(),input_reference_changed=changed,
            hidden_rotor_force_in_unknown=True,metric_force_identified=False,mixed_is_total_thrust=False,
            future_unknown_baseline_not_rotated=True,NN_forecast_blended=False)
        return info


def evaluate_visible(row,horizons):
    c=row.get('coefficient')
    if c is None:return {}
    info=row['visible_force_dynamics'];mode=row['force167_mode'];p=np.asarray(c['position']);v=np.asarray(c['velocity']);a=np.asarray(c['acceleration'])
    n=info['normal'];known=info['components'];gain=info['effective_gain'];result={}
    omega=info['omega'] if mode!='zero' else 0.
    for h in horizons:
        h=float(h);value=p+h*v+kernel(-1/.35,h)*a
        if mode!='zero' and n is not None:
            z=complex(*n);increment=0j
            for key,amplitude in known.items():
                rate=info['log_rates_per_s'][key] if mode!='normal_only' else 0.
                increment+=amplitude*(kernel(rate+1j*omega,h)-.5*h*h)*z
            value+=gain*np.array([increment.real,increment.imag])
        result[str(h)]=value.tolist()
    return result
