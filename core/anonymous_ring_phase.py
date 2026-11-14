"""Persistent anonymous HelixTrack rings, no four-shaft template required.

Past native conics initialize each own chart. Past rotor-region displacement
transports raw observations into that chart. Current early events update the
author core; current late events are scored BEFORE they may update a future
state. Current phase is evidence, transported pose alone is never detection.
"""
from collections import deque
import ctypes as C
import numpy as np
from research.helix_reference.reference_api import AuthorTracker
from angular_phase_candidates import ring_phase_candidates,ring_phase_samples
from joint_phase_bank import usable
from joint_ring_phase import projected_geometry
from ellipse_visibility import ellipse_fov_clipped


class SnapshotTracker(AuthorTracker):
    def snapshot(self):
        self.lib.ht_snapshot_v1.argtypes=[C.c_void_p,C.POINTER(C.c_double)];self.lib.ht_snapshot_v1.restype=C.c_int
        a=np.zeros(21,np.float64);status=self.lib.ht_snapshot_v1(self.handle,a.ctypes.data_as(C.POINTER(C.c_double)))
        if status:raise RuntimeError(f'Author snapshot failed:{status}')
        return dict(first_time_us=float(a[0]),state_time_us=float(a[1]),phase_rad=float(a[2]),omega=float(a[3]),alpha=float(a[4]),H=a[5:14].reshape(3,3,order='F'),q=a[14:20],sign=int(a[20]))


def motion_velocity(row,fallback=None):
    motion=row.get('motion') or {};shift=motion.get('estimate') or {};dt=motion.get('dt_s')
    if dt and motion.get('current_observation') and shift.get('score',0)>0 and not shift.get('peak_at_search_boundary'):
        return np.array([shift['dx'],shift['dy']])/dt
    return np.zeros(2) if fallback is None else np.asarray(fallback,float)


def chart_points(events,seed,shift,velocity,t):
    p=np.c_[events['x'],events['y']]+(t-events['t']/1e6)[:,None]*velocity-shift-[seed['cx'],seed['cy']]
    a=np.deg2rad(seed['angle']);co,si=np.cos(a),np.sin(a)
    normalized=np.c_[(co*p[:,0]+si*p[:,1])/(seed['major']/2),(-si*p[:,0]+co*p[:,1])/(seed['minor']/2)]
    return np.rint(128+80*normalized).astype(np.int32)


def sample(points,H):
    hp=np.c_[points,np.ones(len(points))]@np.linalg.inv(H).T;ok=np.isfinite(hp).all(1)&(abs(hp[:,2])>1e-9)
    q=np.zeros((len(points),2));q[ok]=hp[ok,:2]/hp[ok,2,None]
    radius=np.linalg.norm(q,axis=1);radius[~ok]=np.inf
    theta=np.arctan2(q[:,1],q[:,0]);weight=np.exp(-.5*((radius-1)/.15)**2)
    return radius,theta,weight


def frozen_score(state,events,points,mask,order):
    radius,angle,weight=sample(points,state['H']);keep=mask&(radius>=.2)&(radius<=1.5)
    if keep.sum()<32:return dict(events=int(keep.sum()),concentration=0.,sectors=0),keep
    age=(events['t'][keep]-state['state_time_us'])*1e-6
    phase=state['phase_rad']+state['omega']*age+.5*state['alpha']*age**2
    residual=phase-order*state['sign']*angle[keep];z=np.sum(weight[keep]*np.exp(1j*residual))/max(weight[keep].sum(),1e-12)
    other=np.sum(weight[keep]*np.exp(1j*(phase+order*state['sign']*angle[keep])))/max(weight[keep].sum(),1e-12)
    sectors=len(np.unique(np.floor((angle[keep&(weight>=.1)]+np.pi)*12/(2*np.pi)).astype(int)%12))
    return dict(events=int(keep.sum()),concentration=float(abs(z)),complex_mean=[float(z.real),float(z.imag)],opposite_power=float(abs(other)**2),sectors=sectors),keep


class AnonymousRingPhase:
    max_missing_s=.12
    def __init__(self,author_dll):
        self.dll=author_dll;self.tracks={};self.history=deque();self.next_id=0;self.last_t=None;self.last_past_t=None;self.last_motion=np.zeros(2)

    @staticmethod
    def transport(track,past,t):
        old=track['anchor'];body=past['body'];v=motion_velocity(past,track['velocity'])
        if body.get('detected') and not body.get('center_provisional') and body.get('center') is not None:
            shift=np.asarray(body['center'])-old+v*(t-past['t'])
        else:shift=track['last_shift']+v*(t-track['chart_time'])
        return shift,v

    @staticmethod
    def feed(track,e,xy,mask,*,solve=True):
        ids=np.flatnonzero(mask&(e['t']>track['last_event_us']))
        if not len(ids):return
        out=track['core'].feed(xy[ids,0],xy[ids,1],e['t'][ids],e['p'][ids],solve_pose=False)
        track['last_event_us']=int(e['t'][ids[-1]]);track['pending_pose']+=out['accepted_events']
        # Preserve the existing author2000-event requirement. Sparse windows
        # accumulate across time instead of repeatedly restarting below it.
        if solve and track['pending_pose']>=2000:
            track['core'].feed([],[],[],[],solve_pose=True);track['pending_pose']=0

    def birth(self,past,events,t):
        body=past['body'];born=[]
        if not body.get('detected') or body.get('center_provisional') or body.get('center') is None:return born
        v=motion_velocity(past,self.last_motion);self.last_motion=v
        for g in past['geometry'].get('rotors',[]):
            if not usable(g) or g.get('association_ambiguous') or g.get('shared_arc_recovered'):continue
            seed=dict(g);seed['cx']+=.010*v[0];seed['cy']+=.010*v[1]
            if ellipse_fov_clipped(seed):continue
            duplicate=False
            for track in self.tracks.values():
                shift,_=self.transport(track,past,past['t']);s=track['seed'];centre=np.array([s['cx'],s['cy']])+shift
                if np.linalg.norm(centre-[seed['cx'],seed['cy']])<.3*max(seed['major'],s['major']):duplicate=True
            if duplicate or len(self.tracks)>=4:continue
            control=ring_phase_candidates(events,seed,past['t'],v);m=control.get('angular_harmonic');sf=control.get('candidate_signed_phase_hz')
            if not m or not sf:continue
            proxy=2*abs(sf)/m
            if not 60<=proxy<=1200:continue
            data,_=ring_phase_samples(events,seed,past['t'],v)
            if data is None:continue
            theta=data['theta'];ww=data['weight'];ee=data['events'];first=int(ee['t'][0])
            z=np.sum(ww*np.exp(1j*(m*theta-2*np.pi*sf*(ee['t']-first)*1e-6)));sign=1 if sf>0 else -1
            core=SnapshotTracker(abs(sf)*60,blades=m,dll_path=self.dll,signed_coordinates=True,rotation_sign=sign,initial_phase_rad=sign*np.angle(z))
            track=dict(core=core,seed=seed,anchor=np.asarray(body['center'],float),velocity=v.copy(),birth=past['t'],seen=past['t'],
                streak=0,last_shift=np.zeros(2),chart_time=past['t'],last_event_us=-1,pending_pose=0,order=m)
            xy=chart_points(events,seed,np.zeros(2),v,past['t']);radius,_,_=sample(xy,np.array([[80.,0,128],[0,80,128],[0,0,1.]]));inside=(radius>=.2)&(radius<=1.5)
            self.feed(track,events,xy,inside);i=f'anonymous:{self.next_id}';self.next_id+=1;self.tracks[i]=track;born.append(i)
        return born

    def step(self,row,events,*,return_support=False):
        t=row['t'];end=round(t*1e6);mid=end-33333+33333//2
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing native cutoff times required')
        self.last_t=t
        for i in list(self.tracks):
            if t-self.tracks[i]['seen']>self.max_missing_s+1e-8:self.tracks.pop(i)['core'].close()
        pasts=[r for r in self.history if r['row']['t']<=mid*1e-6+1e-9]
        records=[];support=np.zeros(len(events),bool);born=[]
        if pasts:
            old=pasts[-1];past=old['row']
            if self.last_past_t!=past['t']:
                shift=past['body'].get('measurement_reference_shift_xy')
                if shift is not None:
                    for track in self.tracks.values():track['anchor']+=np.asarray(shift)
                born=self.birth(past,old['events'],t);self.last_past_t=past['t']
            # Event assignment is spatial and uses the previous snapshots only.
            prepared={}
            for i,track in self.tracks.items():
                shift,v=self.transport(track,past,t);xy=chart_points(events,track['seed'],shift,v,t)
                state=track['core'].snapshot();radius,_,_=sample(xy,state['H']);prepared[i]=(shift,v,xy,radius)
            ids=list(prepared);dist=np.array([abs(prepared[i][3]-1) for i in ids]) if ids else np.zeros((0,len(events)))
            closest=np.argmin(dist,axis=0) if ids else np.zeros(len(events),int);sd=np.sort(dist,axis=0)
            margin=sd[1]-sd[0] if len(ids)>1 else np.full(len(events),np.inf)
            for index,i in enumerate(ids):
                track=self.tracks[i];shift,v,xy,radius=prepared[i];m=track['order']
                keep=(closest==index)&(margin>=.1)&(radius>=.2)&(radius<=1.5)&(events['t']>track['last_event_us'])
                early=keep&(events['t']<mid);late=keep&(events['t']>=mid)
                rec=dict(track_id=i,qualified=False,current_force_admitted=False,source_initialization_t=track['birth'],
                    geometry_prior_source_t=past['t'],no_four_shaft_template_required=True,anonymous_track_not_motor_number=True)
                if min(int(early.sum()),int(late.sum()))<32:records.append(dict(rec,reason='few_unique_native_halves'));continue
                self.feed(track,events,xy,early);frozen=track['core'].snapshot()
                tr,_=frozen_score(frozen,events,xy,early,m);va,valid_late=frozen_score(frozen,events,xy,late,m)
                agreement=contrast=None;qualified=False
                if tr.get('complex_mean') is not None and va.get('complex_mean') is not None:
                    a=complex(*tr['complex_mean']);b=complex(*va['complex_mean']);agreement=float(np.real(b*np.conj(a))/max(abs(a),1e-12));power=abs(b)**2;contrast=float((power-va['opposite_power'])/max(power+va['opposite_power'],1e-12))
                    qualified=bool(abs(a)>=.5 and abs(b)>=.5 and agreement>=.5 and contrast>=.5 and va['sectors']>=9)
                try:
                    geometry=projected_geometry(track['seed'],frozen['H']);geometry['cx']+=shift[0];geometry['cy']+=shift[1];geometry['projected_shaft_xy']=(np.asarray(geometry['projected_shaft_xy'])+shift).tolist();geometry['fov_clipped']=ellipse_fov_clipped(geometry)
                except (ValueError,np.linalg.LinAlgError):geometry=dict(invalid=True,fov_clipped=True)
                qualified=qualified and not geometry['fov_clipped']
                frequency=2*abs(frozen['omega']+frozen['alpha']*(end-frozen['state_time_us'])*1e-6)/(2*np.pi*m)
                qualified=bool(qualified and np.isfinite(frequency) and frequency>0)
                fresh=t-track['seen']>=.016667-1e-8
                if qualified:
                    consecutive=t-track['seen']<=.040001;track['streak']=track['streak']+int(fresh) if consecutive else 1
                    track['seen']=t;self.feed(track,events,xy,late)
                admitted=bool(qualified and fresh and track['streak']>=2)
                if admitted:
                    rr,aa,ww=sample(xy,frozen['H']);age=(events['t']-frozen['state_time_us'])*1e-6
                    phase=frozen['phase_rad']+frozen['omega']*age+.5*frozen['alpha']*age**2-m*frozen['sign']*aa
                    support|=keep&(rr>=.2)&(rr<=1.5)&(np.cos(phase)>0)
                track['velocity']=v;track['last_shift']=shift;track['chart_time']=t
                rec.update(qualified=qualified,current_force_admitted=admitted,proxy_twofold_hz=float(frequency) if qualified else None,
                    consecutive_confirmations=track['streak'],reason='persistent_author_late_confirmed' if qualified else 'current_phase_not_confirmed',
                    result=dict(control=dict(angular_harmonic=m),arms=dict(joint_pose=dict(qualified=qualified,inferred_geometry=geometry,train=tr,validation=va,phase_agreement=agreement,direction_contrast=contrast))),
                    snapshot=dict(time_us=frozen['state_time_us'],omega=frozen['omega'],alpha=frozen['alpha']),
                    late_events_update_future_state_only=True,failed_early_updates_retained_in_unconfirmed_tracker=True)
                records.append(rec)
        observation={k:row[k] for k in ('t','body','geometry','structure','motion')};self.history.append(dict(row=observation,events=events.copy()))
        while self.history and t-self.history[0]['row']['t']>.12:self.history.popleft()
        meta=dict(source='anonymous_persistent_author_ring',records=records,qualified_current_rings=sum(r['current_force_admitted'] for r in records),
            no_four_shaft_template_required=True,current_support_is_native_event_subset=True,periodic_pixel_status_is_not_claimed=True,
            current_geometry_is_conditioned_on_phase_fit=True,mechanical_RPM_or_thrust_truth=False,hidden_frequencies_added=0,born_tracks=born,
            current_support_events=int(support.sum()),late_validation_never_updates_its_own_frozen_model=True)
        return (dict(t=t,meta=meta),support) if return_support else dict(t=t,meta=meta)

    def close(self):
        for track in self.tracks.values():track['core'].close()
        self.tracks.clear()
