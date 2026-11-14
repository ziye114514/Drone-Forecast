"""Automatic native-event phase initialization; no saved author outputs.

An unconfirmed spectral peak is ONLY a search prior. It is never published as
a measured frequency. Two later independently validated windows are required
by ContinuousRotorPhase before a current force contribution is admitted.
"""
from collections import deque
import numpy as np
from joint_phase_bank import usable
from joint_ring_phase import projected_geometry
from angular_phase_candidates import ring_phase_candidates
from continuous_rotor_phase import ContinuousRotorPhase,velocity


def discover(row,events,bank):
    t=row['t'];model=np.asarray(row['structure'].get('model_centers') or [],float)
    empty=dict(t=t,meta=dict(records=[],source='native_geometry_and_event_discovery_only'))
    if model.shape!=(4,2):return empty
    v=velocity(row);model=model+.010*v;records=[];H=np.array([[80.,0.,128.],[0.,80.,128.],[0.,0.,1.]])
    for index,g in enumerate(row['geometry'].get('rotors',[])):
        if not usable(g):continue
        seed=dict(g);seed['cx']+=.010*v[0];seed['cy']+=.010*v[1]
        distance=np.linalg.norm(model-[seed['cx'],seed['cy']],axis=1);order=np.argsort(distance);slot=int(order[0])
        if distance[slot]>max(8.,.3*seed['major']) or distance[order[1]]-distance[slot]<3.:continue
        track=bank.tracks.get(slot)
        if track is not None and t-track['seen']<=bank.maximum_missing_s:continue
        control=ring_phase_candidates(events,seed,t,v)
        m=control.get('angular_harmonic');f=control.get('candidate_signed_phase_hz')
        if not m or not f:continue
        proxy=2*abs(float(f))/m
        if not np.isfinite(proxy) or not 60<=proxy<=1200:continue
        geometry=projected_geometry(seed,H)
        if geometry['fov_clipped']:continue
        records.append(dict(track_id=f'native_prior:{t}:{index}',seed=seed,qualified=False,
            current_force_admitted=False,initialization_prior_only=True,proxy_twofold_hz=proxy,
            result=dict(control=control,arms=dict(joint_pose=dict(qualified=False,inferred_geometry=geometry,
                train=dict(concentration=control['discovery_concentration']),
                validation=dict(concentration=control['validation_concentration']),
                author_state=dict(H=H.tolist(),alpha=0.),author_core_used=False)))))
    return dict(empty,meta=dict(empty['meta'],records=records))


class NativeContinuousPhase:
    def __init__(self,dll,*,retain_phase=True):
        self.bank=ContinuousRotorPhase(dll,retain_phase=retain_phase,filter_policy='cohelix_dynamics');self.history=deque()

    def step(self,row,events):
        t=row['t'];mid=(round(t*1e6)-33333+33333//2)/1e6
        eligible=[r for r in self.history if r['row']['t']<=mid+1e-9]
        if eligible:
            old=eligible[-1];past_period=discover(old['row'],old['events'],self.bank)
            u,q,meta=self.bank.step(row,events,old['row'],past_period,old['events'])
        else:u,q,meta=self.bank.step(row,events,None,None,None)
        # Store only the observations used by this module; no model state or
        # future reference can silently enter its initializer.
        observation={k:row[k] for k in ('t','body','geometry','structure','motion')}
        self.history.append(dict(row=observation,events=events.copy()))
        while self.history and t-self.history[0]['row']['t']>.12:self.history.popleft()
        meta.update(initialization_requires_author_period_file=False,initialization_is_not_current_frequency_measurement=True)
        return dict(t=t,u=None if u is None else u.tolist(),quality=q,meta=meta)
