"""Native rotor increments added to the preserved legacy NN motion backbone.

Old p/v/flow/validity are fixed; no second trajectory is estimated. Visible
force contributions use an explicit zero-origin gauge so changing which
rotors are observed does not manufacture thrust. Hidden baselines stay in d.
"""
import copy
import numpy as np
from rotor_thrust_bank import RotorThrustBankProxy
from endpoint_force_balance import EndpointForceBalance
from normal_turn_forecast import NormalTurnForecast
from adaptive_force_memory import AdaptiveForceMemory
from axis_amplitude_bridge import current_axis_qualified


class ObservedForceGauge:
    def __init__(self):self.previous={};self.offset=np.zeros(2)
    def step(self,u,meta):
        current={k:np.asarray(v,float) for k,v in meta.get('observed_components',{}).items()} if u is not None else {}
        old=set(self.previous);new=set(current);delta=np.zeros(2)
        for key in new-old:delta+=current[key]
        for key in old-new:delta-=self.previous[key]
        self.offset+=delta;self.previous=current
        value=None if u is None else np.asarray(u)-self.offset
        return value,dict(visibility_gauge_shift=delta.tolist(),gauge_offset=self.offset.tolist(),
            observed_raw_u=None if u is None else np.asarray(u).tolist(),
            hidden_force_in_unknown=True,source_change_not_force_impulse=True)


class PreservedNNRotorResidual:
    def __init__(self,mode):
        self.mode=mode
        if mode not in ('legacy_exact','native_pose','native_four','native_visible','native_visible_no_gain','native_evidence_only'):
            raise ValueError(mode)
        self.force=EndpointForceBalance(gain_signal_to_noise=0.)
        self.force.evidence_gain=True
        self.normal=NormalTurnForecast();self.memory=AdaptiveForceMemory(extend_calibrated=True)
        self.proxy=RotorThrustBankProxy(vector_pool=True);self.gauge=ObservedForceGauge()

    def step(self,old,observation):
        t=old['t'];c=copy.deepcopy(old['coefficient']);motion=old.get('image_motion') or {}
        native=self.mode!='legacy_exact';gauge_info=None;meta=None
        if native:
            s=observation['structure']
            u,q,meta=self.proxy.step(t,s,observation['geometry'].get('rotors',[]),observation['rates'],observation['pooled'])
            if self.mode=='native_pose' or not observation.get('current_pose') or self.mode=='native_four' and not(s.get('mode')=='four' and len(meta.get('current_model_ids',[]))==4 and meta.get('rotor_bank_used')):
                u=None;q=0.;meta=dict(meta,observed_components={})
            u,gauge_info=self.gauge.step(u,meta)
            # Matched-availability ablation: retain actual periodic/pose gates,
            # d adaptation and memory, but remove the known-input value. This
            # separates useful periodic force from a mere "enable d" flag.
            if self.mode=='native_evidence_only' and u is not None:u=np.zeros(2)
            # Mixed periodic evidence cannot identify k. Keep its actual input
            # history for fixed-k disturbance updates, rather than disabling d
            # learning whenever gain identification is forbidden.
            self.force.input_learning_current=True
            self.force.gain_learning_enabled=bool(meta.get('learning_input_current',False) and self.mode not in ('native_visible_no_gain','native_evidence_only'))
        else:
            s=old['structure'];u=old['force'].get('input_u');q=.35
            self.force.input_learning_current=(old.get('axis_amplitude_bridge') or {}).get('learning_input_current',True)
        self.force.observe_flow(motion.get('observation'),bool((motion.get('acceleration') or {}).get('applied')))
        self.force.observe_axis_motion(s,motion)
        p,f=self.force.step(t,c,old.get('force_observed_position'),u,q)
        trend=self.normal.observe(t,s,c,motion);p,n=self.normal.correct(p,f,trend)
        p,memory=self.memory.step(t,c,old.get('force_observed_position'),p,f,n,old['valid'],
            bool((motion.get('acceleration') or {}).get('applied')),current_axis_qualified(s))
        row=dict(t=t,body=copy.deepcopy(old['body']),coefficient=c,valid=old['valid'],force=f,physical=p,
            structure=s,normal_turn_forecast=n,adaptive_force_memory=memory,
            native_input_selection=meta,observed_force_gauge=gauge_info,mode=self.mode,
            original_NN_position_velocity_and_validity_preserved=True)
        return row
