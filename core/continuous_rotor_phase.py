"""Persistent per-source phase in a shared five-circle reference plane.

Initialization uses only past automatically confirmed ring observations.
Current discovery events update a retained phase/rate/acceleration state;
the late half is frozen validation, then may update FUTURE internal state.
No trajectory references, NN, physical motor names or hidden RPM are supplied.
"""
import numpy as np
from continuous_phase_filter import initial,feed,predict,frozen_validation
from joint_ring_phase import projected_geometry
from axis_amplitude_bridge import current_axis_qualified


IDENTITY_ELLIPSE=dict(cx=128.,cy=128.,major=160.,minor=160.,angle=0.)


def image_chart(record):
    seed=record['seed'];a=np.deg2rad(seed['angle']);B=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])@np.diag([seed['major']/160,seed['minor']/160])
    M=np.eye(3);M[:2,:2]=B;M[:2,2]=[seed['cx'],seed['cy']]-B@np.array([128.,128.])
    return M@np.asarray(record['result']['arms']['joint_pose']['author_state']['H'])


def velocity(row):
    mo=row.get('motion') or {};s=mo.get('estimate') or {};dt=mo.get('dt_s')
    if dt and mo.get('current_observation') and not s.get('peak_at_search_boundary') and s.get('score',0)>0:
        return np.array([s['dx'],s['dy']])/dt
    return np.zeros(2)


class ContinuousRotorPhase:
    maximum_missing_s=.12
    def __init__(self,dll,*,retain_phase=True,filter_policy='mixture'):
        self.dll=dll;self.retain_phase=retain_phase;self.reference=None;self.epoch=None
        self.filter_policy=filter_policy
        self.tracks={};self.priors={};self.next_id=0;self.last_t=None

    def transform(self,row,t):
        hubs=np.asarray(row['structure'].get('model_centers') or [],float)
        if hubs.shape!=(4,2):return None
        v=velocity(row);hubs=hubs+v*(t-row['t']+.010)
        if self.reference is None:self.reference=hubs.copy()
        D=np.c_[hubs,np.ones(4)];T=np.linalg.lstsq(D,self.reference,rcond=None)[0]
        singular=np.linalg.svd(T[:2],compute_uv=False)
        if np.linalg.det(T[:2])<=0 or singular.min()<.1 or singular.max()>10 or singular.max()/singular.min()>20:return None
        G=np.eye(3);G[:2,:]=T.T
        return G,v,hubs

    @staticmethod
    def points(events,t,G,v):
        xy=np.c_[events['x'],events['y']]+(t-events['t']/1e6)[:,None]*v
        return np.c_[xy,np.ones(len(xy))]@G.T

    @staticmethod
    def sample(points,J):
        q=points@np.linalg.inv(J).T;finite=np.isfinite(q).all(1)&(abs(q[:,2])>1e-9)
        u=np.zeros((len(q),2));u[finite]=q[finite,:2]/q[finite,2,None]
        radius=np.linalg.norm(u,axis=1);radius[~finite]=np.inf
        angle=np.arctan2(u[:,1],u[:,0]);weight=np.exp(-.5*((radius-1)/.15)**2)
        return radius,angle,weight

    def birth(self,row,period,events):
        tr=self.transform(row,row['t'])
        if tr is None:return []
        G,v,hubs=tr;born=[];choices={}
        for record in (period.get('meta') or {}).get('records',[]):
            if not (record.get('qualified') or record.get('initialization_prior_only')) or 'track_id' not in record:continue
            arm=record['result']['arms']['joint_pose'];g=arm['inferred_geometry']
            if g.get('fov_clipped') or g['minor']/g['major']<=.16:continue
            p=np.asarray(g.get('projected_shaft_xy',[g['cx'],g['cy']]))
            distances=np.linalg.norm(hubs-p,axis=1);order=np.argsort(distances);slot=int(order[0])
            if distances[slot]>max(8.,.3*g['major']) or distances[order[1]]-distances[slot]<3.:continue
            if slot in self.tracks and row['t']-self.tracks[slot]['seen']<=self.maximum_missing_s:continue
            quality=min(arm['train']['concentration'],arm['validation']['concentration'])
            if slot not in choices or quality>choices[slot][0]:choices[slot]=(quality,record)
        if not choices:return []
        points=self.points(events,row['t'],G,v)
        for slot,(_,record) in choices.items():
            J=G@image_chart(record);radius,angle,weight=self.sample(points,J)
            valid=(radius>=.2)&(radius<=1.5)
            if valid.sum()<64:continue
            control=record['result']['control'];m=int(control['angular_harmonic']);sign=1 if control['candidate_signed_phase_hz']>0 else -1
            omega=2*np.pi*float(record['proxy_twofold_hz'])*m/2
            age=(events['t'][valid]-round(row['t']*1e6))/1e6
            z=np.sum(weight[valid]*np.exp(1j*(m*sign*angle[valid]-omega*age)))
            # The accepted discovery supplies a rate, not a separately
            # qualified angular acceleration. Let persistent events learn it.
            alpha=0. if self.filter_policy=='cohelix_dynamics' else float(record['result']['arms']['joint_pose']['author_state']['alpha'])
            state=initial(round(row['t']*1e6),np.angle(z),omega,alpha)
            self.tracks[slot]=dict(id=f'continuous:{self.epoch}:{self.next_id}:{slot}',J=J,state=state,
                seen=row['t'],streak=0,order=m,sign=sign,birth_t=row['t'],
                source_geometry='past_native_discovery_prior' if record.get('initialization_prior_only') else 'past_late_confirmed_ring_in_shared_reference')
            self.next_id+=1;born.append(slot);self.priors[slot]=J.copy()
            # Other model points supply only exclusion/search hypotheses. They
            # never create a frequency or a current rotor observation.
            for j in range(4):
                if j in self.priors:continue
                shift=np.eye(3);shift[:2,2]=self.reference[j]-self.reference[slot];self.priors[j]=shift@J
        return born

    def step(self,row,events,past_row,past_period,past_events):
        t=row['t'];mid=(round(t*1e6)-33333+33333//2)/1e6
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing native cutoffs required')
        self.last_t=t
        empty=dict(source='continuous_shared_reference_phase',records=[],observed_components={},qualified_current_rings=0,
            learning_input_current=False,input_sigma_proxy=0.,hidden_frequencies_measured=0,
            metric_thrust_measured=False,mechanical_RPM_measured=False,physical_motor_identity_required=False)
        if past_row is None:return None,0.,dict(empty,reason='await_past_automatic_initialization')
        if past_row['t']>mid+1e-9:raise ValueError('Shared geometry/initialization must precede current validation')
        epoch=past_row['structure'].get('geometry_epoch')
        if self.epoch!=epoch:
            self.reference=None;self.tracks.clear();self.priors.clear();self.epoch=epoch
        self.tracks={i:r for i,r in self.tracks.items() if t-r['seen']<=self.maximum_missing_s+1e-8}
        born=self.birth(past_row,past_period,past_events)
        tr=self.transform(past_row,t)
        if tr is None or not self.tracks:return None,0.,dict(empty,reason='no_conditioned_shared_geometry_or_phase',born_slots=born)
        G,v,_=tr;end=round(t*1e6);begin=end-33333;mid_us=begin+33333//2
        events=events[(events['t']>=begin)&(events['t']<end)];points=self.points(events,t,G,v)
        samples={i:self.sample(points,J) for i,J in self.priors.items()}
        slots=sorted(samples);dist=np.array([abs(samples[i][0]-1) for i in slots]);sorted_dist=np.sort(dist,axis=0)
        best=np.asarray(slots)[np.argmin(dist,axis=0)];margin=sorted_dist[1]-sorted_dist[0] if len(slots)>1 else np.full(len(events),np.inf)
        records=[];components={};newly_confirmed=[]
        axis=row['structure'].get('axis_xy');n=np.asarray(axis,float) if axis is not None else np.zeros(0)
        axis_good=current_axis_qualified(row['structure']) and n.shape==(2,) and np.isfinite(n).all() and np.linalg.norm(n)>1e-8
        if axis_good and n[1]>0:n=-n
        for slot,track in sorted(self.tracks.items()):
            radius,angles,weights=samples[slot]
            keep=(best==slot)&(margin>=.1)&(radius>=.2)&(radius<=1.5)
            # A final short RAW packet may overlap most of the rolling window.
            # Events already committed to this source are never fed twice.
            keep&=events['t']>track['state'][0]
            tt=events['t'][keep];aa=angles[keep];ww=weights[keep];early=tt<mid_us
            rec=dict(track_id=track['id'],model_slot=slot,qualified=False,current_force_admitted=False,
                state_retained_across_windows=self.retain_phase,source_initialization_t=track['birth_t'],
                geometry_source_t=past_row['t'],validation_half_disjoint_from_previous_confirmation=True)
            if min(int(early.sum()),int((~early).sum()))<32:
                records.append(dict(rec,reason='few_unique_current_events'));continue
            state=track['state'].copy();m=track['order'];sign=track['sign']
            if not self.retain_phase:
                age=(tt[early]-state[0])*1e-6;omega=float(state[2]+state[3]*age[0]);dt=(tt[early]-tt[early][0])*1e-6
                z=np.sum(ww[early]*np.exp(1j*(m*sign*aa[early]-omega*dt)))
                state=initial(tt[early][0],np.angle(z),omega,0.)
            fitted=feed(state,tt[early],m*sign*aa[early],ww[early],dll=self.dll,policy=self.filter_policy)
            validation=frozen_validation(fitted,tt,aa,ww,early,order=m,sign=sign)
            Jimage=np.linalg.inv(G)@track['J']
            try:geometry=projected_geometry(IDENTITY_ELLIPSE,Jimage)
            except (ValueError,np.linalg.LinAlgError):geometry=dict(invalid=True,fov_clipped=True)
            age=(end-fitted[0])*1e-6;f=float(2*abs(fitted[2]+fitted[3]*age)/(2*np.pi*m))
            qualified=bool(validation['qualified'] and not geometry['fov_clipped'] and np.isfinite(f) and f>0)
            if qualified:
                # Commit validation only AFTER scoring; it supplies a prior for
                # a later disjoint window, not a re-scored current measurement.
                track['state']=feed(fitted,tt[~early],m*sign*aa[~early],ww[~early],dll=self.dll,policy=self.filter_policy)
                fresh=t-track['seen']>=.016667-1e-8
                consecutive=t-track['seen']<=.040001;track['streak']=track['streak']+int(fresh) if consecutive else 1
                track['seen']=t;admitted=track['streak']>=2
                admitted=admitted and fresh
                if admitted and axis_good:components[track['id']]=(f*f/360000*n).tolist()
                newly_confirmed.append(slot)
            else:admitted=False
            geometry.update(is_inferred_geometry=True,position_is_not_independent_ground_truth=True,
                current_ellipse_independently_refitted=False,geometry_is_past_model_transported=True)
            rec.update(qualified=qualified,current_force_admitted=admitted,proxy_twofold_hz=f if qualified else None,
                consecutive_confirmations=track['streak'],reason='current_frozen_late_phase_confirmed' if qualified else 'current_phase_not_confirmed',
                result=dict(control=dict(angular_harmonic=m),arms=dict(joint_pose=dict(inferred_geometry=geometry,**validation))),
                frozen_rate_rad_s=float(fitted[2]),frozen_acceleration_rad_s2=float(fitted[3]),
                retained_phase_state_us=float(track['state'][0]),validation_updates_future_internal_state_only=True,
                published_rate_is_frozen_validation_model=True)
            records.append(rec)
        u=np.sum(list(components.values()),axis=0) if components else None
        magnitude=np.sqrt(sum(np.dot(v,v) for v in components.values()))
        meta=dict(empty,records=records,observed_components=components,qualified_current_rings=sum(r['current_force_admitted'] for r in records),
            current_force_components=len(components),input_sigma_proxy=float(.04*magnitude),learning_input_current=bool(components),
            input_uncertainty_calibrated=False,complete_five_circle_required=False,source_set_is_not_physical_heading=True,
            born_slots=born,current_late_confirmed_slots=newly_confirmed,shared_geometry_source_t=past_row['t'],
            all_current_admitted_native_timestamps_preserved=True,hidden_force_owner='joint_unknown_state_d',
            NN_used=False,reason='continuous_native_phase_with_frozen_validation')
        return u,.7 if u is not None else 0.,meta
