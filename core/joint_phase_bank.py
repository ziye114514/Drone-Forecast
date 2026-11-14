"""Causal, optional CPU ring-phase observations; no NN and no hidden RPM.

Past native conics initialize a discovery-only joint fit. Current late events
must confirm it. Retained geometry permits relock, never stale measured force.
This accuracy research implementation is not a real-time performance claim.
"""
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from joint_ring_phase import observe_ring
from ellipse_visibility import ellipse_fov_clipped
from conic_observability import usable


class JointPhaseBank:
    def __init__(self,author_dll,temporal_hypothesis=False):
        self.author_dll=author_dll;self.previous=None;self.history=deque();self.tracks={};self.next_id=0
        self.temporal_hypothesis=temporal_hypothesis

    def step(self,t,events,rotors,motion):
        if self.previous is not None and t<=self.previous['t']:
            raise ValueError('Source cutoffs must increase')
        self.tracks={i:r for i,r in self.tracks.items() if t-r['seen']<=.12+1e-8}
        proposals=[];velocity=np.zeros(2);velocity_t=None
        mid=(round(t*1e6)-33333+33333//2)/1e6
        while self.history and t-self.history[0]['t']>.12+1e-8:self.history.popleft()
        eligible=[r for r in self.history if r['t']<=mid+1e-9]
        if eligible:
            old=eligible[-1];shift=(old['motion'] or {}).get('estimate') or {};dt=(old['motion'] or {}).get('dt_s')
            if dt and old['motion'].get('current_observation') and not shift.get('peak_at_search_boundary') and shift.get('score',0)>0:
                velocity=np.array([shift['dx'],shift['dy']])/dt;velocity_t=old['t']
            # Native20ms counts describe their midpoint. The initial pose for
            # the new window is transported to its cutoff, using only history.
            for r in old['rotors']:
                if not usable(r):continue
                q=dict(r);q['cx']+=velocity[0]*(t-old['t']+.010);q['cy']+=velocity[1]*(t-old['t']+.010)
                if not ellipse_fov_clipped(q):proposals.append((q,old['t'],'past_native_conic'))
            # A previously confirmed pose is a search seed, not a measurement.
            for i,r in self.tracks.items():
                if r['seen']>mid+1e-9:continue
                q=dict(r['ellipse']);q['cx']+=velocity[0]*(t-r['seen']);q['cy']+=velocity[1]*(t-r['seen'])
                if ellipse_fov_clipped(q):continue
                if any(np.linalg.norm([q['cx']-v[0]['cx'],q['cy']-v[0]['cy']])<.3*max(q['major'],v[0]['major']) for v in proposals):continue
                proposals.append((q,r['seen'],'past_confirmed_pose'))
        observations=[]
        for q,source,kind in proposals:
            result=observe_ring(events,q,t,velocity if velocity_t is not None else None,
                author_dll=self.author_dll,geometry_source_end_s=source,velocity_source_end_s=velocity_t)
            selection=dict(temporal_hypothesis_enabled=self.temporal_hypothesis,prior_candidate_tried=False,
                selected='current_discovery',selection_uses_validation=False)
            if self.temporal_hypothesis:
                nearby=[]
                for i,track in self.tracks.items():
                    if track['seen']>mid+1e-9 or 'phase_order' not in track:continue
                    g=track['ellipse'];expected=np.array([g['cx'],g['cy']])+velocity*(t-track['seen'])
                    dist=float(np.linalg.norm(expected-[q['cx'],q['cy']]))
                    scales=np.array([g['major']/q['major'],g['minor']/q['minor']])
                    if scales.min()>=.65 and scales.max()<=1.5 and dist<.3*max(g['major'],q['major']):nearby.append((dist,i))
                nearby.sort()
                if nearby and (len(nearby)==1 or nearby[1][0]-nearby[0][0]>=3.):
                    track=self.tracks[nearby[0][1]]
                    hint=dict(source_t=track['seen'],twofold_hz=track['frequency'],angular_order=track['phase_order'],rotation_sign=track['rotation_sign'])
                    prior=observe_ring(events,q,t,velocity if velocity_t is not None else None,
                        author_dll=self.author_dll,geometry_source_end_s=source,velocity_source_end_s=velocity_t,phase_hint=hint)
                    def discovery_score(candidate):
                        a=candidate.get('arms',{}).get('joint_pose',{});g=a.get('inferred_geometry') or {};train=a.get('train') or {}
                        if not g or g.get('fov_clipped') or g.get('invalid'):return -1.
                        scales=np.array([g['major']/q['major'],g['minor']/q['minor']]);dist=np.linalg.norm([g['cx']-q['cx'],g['cy']-q['cy']])
                        if scales.min()<.65 or scales.max()>1.5 or dist>max(8.,.6*max(g['major'],q['major'])):return -1.
                        return float(train.get('concentration',0.))
                    a,b=discovery_score(result),discovery_score(prior)
                    selection.update(prior_candidate_tried=True,prior_source_t=track['seen'],discovery_scores=[a,b])
                    if b>a:result=prior;selection['selected']='past_frequency_current_discovery_refit'
            arm=result.get('arms',{}).get('joint_pose',{});g=arm.get('inferred_geometry',{})
            qualified=bool(arm.get('qualified'))
            if qualified:
                scales=np.array([g['major']/q['major'],g['minor']/q['minor']])
                dist=np.linalg.norm([g['cx']-q['cx'],g['cy']-q['cy']])
                qualified=bool(scales.min()>=.65 and scales.max()<=1.5 and dist<=max(8.,.6*max(g['major'],q['major'])))
            m=result.get('control',{}).get('angular_harmonic');f=None
            if qualified:
                state=arm['author_state'];age=t-arm['discovery_state_time_s']
                # One common twofold gauge across selected angular orders.
                f=2*abs(state['angular_velocity_rad_s']+state['alpha']*age)/(2*np.pi*m)
                qualified=bool(np.isfinite(f) and f>0)
            observations.append(dict(seed=q,geometry_source=kind,qualified=qualified,temporal_hypothesis_selection=selection,
                proxy_twofold_hz=f,result=result))
        accepted=[j for j,o in enumerate(observations) if o['qualified']]
        ambiguous=set()
        for a,j in enumerate(accepted):
            x=observations[j]['result']['arms']['joint_pose']['inferred_geometry']
            for k in accepted[:a]:
                y=observations[k]['result']['arms']['joint_pose']['inferred_geometry']
                if np.linalg.norm([x['cx']-y['cx'],x['cy']-y['cy']])<.3*max(x['major'],y['major']):ambiguous.update([j,k])
        if len(accepted)>4:ambiguous.update(accepted)
        accepted=[j for j in accepted if j not in ambiguous]
        ids=list(self.tracks);cost=np.full((len(ids),len(accepted)),1e6);mapping={}
        for a,i in enumerate(ids):
            old=self.tracks[i];r=old['ellipse'];expected=np.array([r['cx'],r['cy']])+velocity*(t-old['seen'])
            for b,j in enumerate(accepted):
                g=observations[j]['result']['arms']['joint_pose']['inferred_geometry']
                scales=np.array([g['major']/r['major'],g['minor']/r['minor']]);dist=np.linalg.norm([g['cx']-expected[0],g['cy']-expected[1]])
                gate=max(8.,.6*max(g['major'],r['major']))
                if scales.min()>=.65 and scales.max()<=1.5 and dist<=gate:cost[a,b]=dist/gate
        if ids and accepted:
            aa,bb=linear_sum_assignment(cost);mapping={accepted[b]:ids[a] for a,b in zip(aa,bb) if cost[a,b]<1e5}
        components={};rapid=[];fresh_confirmations=[]
        for j in accepted:
            o=observations[j];g=o['result']['arms']['joint_pose']['inferred_geometry'];f=o['proxy_twofold_hz']
            i=mapping.get(j)
            if i is None:i=self.next_id;self.next_id+=1
            old=self.tracks.get(i);consecutive=old is not None and t-old['seen']<=.040001
            fresh=old is None or t-old.get('confirmation_t',old['seen'])>=.016667-1e-8
            streak=old['streak']+int(fresh) if consecutive else 1
            self.tracks[i]=dict(seen=t,ellipse=g,frequency=f,streak=streak,
                confirmation_t=t if fresh else old.get('confirmation_t',old['seen']),
                phase_order=o['result']['control']['angular_harmonic'],
                rotation_sign=1 if o['result']['control']['candidate_signed_phase_hz']>0 else -1)
            o.update(track_id=i,consecutive_confirmations=streak,current_force_admitted=streak>=2,
                validation_half_disjoint_from_previous_confirmation=fresh)
            if streak<2:continue
            fresh_confirmations.append(fresh)
            ratio=g['minor']/g['major'];angle=np.deg2rad(g['angle'])
            n=np.array([-np.sin(angle),np.cos(angle)])*np.sqrt(max(0.,1-ratio**2))
            if n[1]>0:n=-n
            if abs(np.log(f/old['frequency']))>=.08:rapid.append(i)
            components[f'joint_phase:{i}']=(f*f/360000*n).tolist()
        self.previous=dict(t=t,rotors=rotors,motion=motion or {})
        self.history.append(self.previous)
        u=np.sum(list(components.values()),axis=0) if components else None
        magnitude=np.sqrt(sum(np.dot(v,v) for v in components.values()))
        meta=dict(source='independent_late_confirmed_joint_phase',observed_components=components,
            input_sigma_proxy=float(.04*magnitude),input_uncertainty_calibrated=False,
            learning_input_current=bool(components and not rapid and all(fresh_confirmations)),qualified_current_rings=len(components),
            hidden_frequencies_measured=0,hidden_force_owner='joint_unknown_state_d',
            physical_motor_identity_required=False,mechanical_RPM_measured=False,metric_thrust_measured=False,
            complete_five_circle_required=False,late_qualified_before_temporal_confirmation=len(accepted),
            ambiguous_candidates=sorted(ambiguous),two_consecutive_confirmations_required=True,
            geometry_is_model_conditioned=True,records=observations)
        return u,.7 if u is not None else 0.,meta
