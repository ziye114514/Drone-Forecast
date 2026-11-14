"""Automatic author-assisted initialization only; persistent own phase core.

The author core refines a discovery search prior, not a current measurement.
Its initializer can use the whole PAST native-conic window. Only subsequent
independent late halves can publish measured force contributions.
"""
from collections import deque
import numpy as np
from native_phase_discovery import discover
from continuous_rotor_phase import ContinuousRotorPhase,velocity
from joint_ring_phase import observe_ring


class BootstrappedContinuousPhase:
    def __init__(self,phase_dll,author_dll):
        self.bank=ContinuousRotorPhase(phase_dll,retain_phase=True,filter_policy='cohelix_dynamics')
        self.author_dll=author_dll;self.history=deque()

    def initialize(self,row,events):
        period=discover(row,events,self.bank);t=row['t'];v=velocity(row)
        for record in period['meta']['records']:
            seed=record['seed'];control=record['result']['control'];m=control['angular_harmonic'];sign=1 if control['candidate_signed_phase_hz']>0 else -1
            hint=dict(source_t=t,twofold_hz=record['proxy_twofold_hz'],angular_order=m,rotation_sign=sign)
            result=observe_ring(events,seed,t,v,author_dll=self.author_dll,geometry_source_end_s=t,
                velocity_source_end_s=t,phase_hint=hint,initialization_only=True)
            arm=result.get('arms',{}).get('joint_pose',{});g=arm.get('inferred_geometry') or {}
            if not g or g.get('invalid') or g.get('fov_clipped'):continue
            scales=np.array([g['major']/seed['major'],g['minor']/seed['minor']]);distance=np.linalg.norm([g['cx']-seed['cx'],g['cy']-seed['cy']])
            if scales.min()<.65 or scales.max()>1.5 or distance>max(8.,.6*max(g['major'],seed['major'])):continue
            old=record['result']['arms']['joint_pose'];old_score=min(old['train']['concentration'],old['validation']['concentration'])
            score=min(arm.get('train',{}).get('concentration',0.),arm.get('validation',{}).get('concentration',0.))
            if score<=old_score:continue
            state=arm['author_state'];age=t-arm['discovery_state_time_s'];frequency=2*abs(state['angular_velocity_rad_s']+state['alpha']*age)/(2*np.pi*m)
            if not np.isfinite(frequency) or not 60<=frequency<=1200:continue
            record.update(result=result,proxy_twofold_hz=float(frequency),initialization_refined_by_author=True,
                qualified=False,current_force_admitted=False,initialization_prior_only=True)
        return period

    def step(self,row,events):
        t=row['t'];mid=(round(t*1e6)-33333+33333//2)/1e6;eligible=[x for x in self.history if x['row']['t']<=mid+1e-9]
        if eligible:
            old=eligible[-1];prior=self.initialize(old['row'],old['events']);u,q,meta=self.bank.step(row,events,old['row'],prior,old['events'])
            refinements=sum(r.get('initialization_refined_by_author',False) for r in prior['meta']['records'])
        else:u,q,meta=self.bank.step(row,events,None,None,None);refinements=0
        observation={k:row[k] for k in ('t','body','geometry','structure','motion')};self.history.append(dict(row=observation,events=events.copy()))
        while self.history and t-self.history[0]['row']['t']>.12:self.history.popleft()
        meta.update(initialization_requires_saved_author_periods=False,author_used_only_for_past_search_prior=True,
            author_bootstrap_refinements=refinements,initialization_is_not_current_frequency_measurement=True)
        return dict(t=t,u=None if u is None else u.tolist(),quality=q,meta=meta)
