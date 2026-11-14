"""Reusable LOCAL-only author-core observer, initialized by CURRENT geometry.

Uses no recordings, old outputs, labels or logged geometry. This is an auxiliary
observation: it never changes the main pipeline's motion/forecast states.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,time
import numpy as np
from scipy.optimize import linear_sum_assignment
from reference_api import AuthorTracker,OUT
from sign_bank import SignBank
from roi_bounds import prefilter,circle_bounds
from structure_prior import image_ellipse,missing_ellipse
import native79
from native_axis141 import measure_current


class RotorPhaseObserver:
    def __init__(self,allow_current_reanchor=False,allow_partial_reanchor=False,allow_shadow_recovery=False):
        self.dll=OUT/'adaptive_signed/build/Release/helix_reference.dll'
        self.engine=native79.engine(1280,720,'strict7');self.buffer=self.engine.get_empty_output_buffer()
        self.pool=ThreadPoolExecutor(max_workers=4);self.banks={};self.rotors=None
        self.anchor_body=None;self.regions={};self.last_body_t=None
        self.last_seen={};self.last_guided={};self.last_frequency={};self.opposite={}
        self.initialized_t=None;self.closed=False
        self.proxy_reference=None;self.proxy_signature=None
        self.allow_current_reanchor=allow_current_reanchor;self.last_complete_t=None
        self.current_four_streak=0;self.epoch=0;self.last_initialization_t=None
        self.allow_partial_reanchor=allow_partial_reanchor
        self.last_good_centers={};self.slot_epoch={};self.proxy_epoch=0
        self.allow_shadow_recovery=allow_shadow_recovery;self.shadows={};self.shadow_last_spawn={}
        self.output_source_signature=None

    def protocol(self):
        own=Path(__file__).parent
        paths=[own/f for f in ['phase_state.py','reference_api.py','sign_bank.py','roi_bounds.py','structure_prior.py']]
        paths.append(own.parent/'stage141/native_axis141.py')
        return dict(type='local original HelixTrack author-core auxiliary observer',
            initial_geometry='CURRENT bounded four independent ellipses, current model and NN body only',
            period_source='Installed SDK strict7, current unique-pixel mode and common-mode rejection',
            translation='CURRENT NN body displacement; current motion estimate interpolates within native20ms',
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            DLL_sha256=hashlib.sha256(self.dll.read_bytes()).hexdigest(),
            four_independent_rotor_threads=True,current_period_required=True,
            allow_current_reanchor=self.allow_current_reanchor,reanchor_loss_s=.3,
            minimum_fresh_conic_streak=2,reanchor_cooldown_s=.5,
            allow_partial_reanchor=self.allow_partial_reanchor,
            allow_shadow_recovery=self.allow_shadow_recovery,
            shadow_policy='Keep primary unchanged; current-supported shadow may fill ONLY primary-invalid slots; reject duplicate centers',
            body_axis_fallback='Reuse141 stateless current periodic-pixel line; unsigned projected axis, no RPM/force',
            partial_association='SciPy Hungarian current conics vs current/past confirmed body-relative conic centers',
            partial_max_history_s=.5,partial_min_unique_margin_major=.25,
            only_observation=True,changes_motion_prediction=False,assumed_B=2,
            true_mechanical_RPM=False,true_force=False,redistribution=False)

    def factory(self,rpm,blades):
        return AuthorTracker(rpm,blades,dll_path=self.dll,canvas=(256,256,128,128,80),signed_coordinates=True)

    def reset(self):
        for bank in self.banks.values():bank.close()
        self.banks={};self.rotors=None;self.anchor_body=None;self.initialized_t=None
        self.proxy_reference=None;self.proxy_signature=None
        self.last_complete_t=None
        self.last_good_centers={}
        for track in self.shadows.values():track['bank'].close()
        self.shadows={};self.shadow_last_spawn={};self.output_source_signature=None

    def initialize(self,rotors,t,body,q):
        seeds=[]
        for r in rotors:
            a=np.deg2rad(r['angle']);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            shape=R@np.diag([(r['major']/2)**2,(r['minor']/2)**2])@R.T
            margin=np.sqrt(np.diag(shape));c=np.array([r['cx'],r['cy']])
            seed=native79.summarize(q,np.r_[c-margin-4,c+margin+4],t,1280,720)
            if not seed.get('qualified') or not seed.get('frequency_hz'):return False
            seeds.append(seed['frequency_hz'])
        self.reset()
        self.rotors=[dict(r) for r in rotors];self.anchor_body=np.asarray(body['center'],float);self.initialized_t=t
        self.last_initialization_t=t;self.last_complete_t=t;self.epoch+=1
        self.proxy_epoch+=1
        for i,seed in enumerate(seeds):
            self.banks[i]=SignBank(seed,2,self.factory)
            self.regions[i]=[np.array([[80.,0,128],[0,80,128],[0,0,1]])]
            self.last_frequency[i]=seed;self.last_seen[i]=t;self.last_guided[i]=t
            self.last_good_centers[i]=np.array([rotors[i]['cx'],rotors[i]['cy']]);self.slot_epoch[i]=0
        centers=np.asarray([[r['cx'],r['cy']] for r in rotors]);z=centers-centers.mean(axis=0)
        order=np.argsort(np.arctan2(z[:,1],z[:,0]))
        self.opposite={int(order[k]):int(order[(k+2)%4]) for k in range(4)}
        return True

    def rotor_step(self,i,t,e,xy,shift,q,track=None):
        r=self.rotors[i] if track is None else track['rotor']
        regions=self.regions[i] if track is None else track['regions']
        bank=self.banks[i] if track is None else track['bank']
        a=np.deg2rad(r['angle']);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        box=prefilter(xy,regions,r,R);local=xy[box]-[r['cx'],r['cy']];ev=e[box]
        patch=np.rint(128+80*(local@R)/np.array([r['major']/2,r['minor']/2]));hp=np.c_[patch,np.ones(len(ev))]
        keep=np.zeros(len(ev),bool)
        for H in regions:
            u=hp@np.linalg.inv(H).T;rad=np.linalg.norm(u[:,:2]/np.maximum(u[:,2,None],1e-6),axis=1)
            keep|=(u[:,2]>1e-6)&(rad>=.2)&(rad<=1.5)
        patch=patch[keep];ev=ev[keep]
        packed=np.empty(len(ev),dtype=[('x','<i4'),('y','<i4'),('p','<i2'),('t','<i8')])
        packed['x']=patch[:,0];packed['y']=patch[:,1];packed['p']=ev['p'];packed['t']=ev['t']
        M=R@np.diag([r['major']/2,r['minor']/2])/80;W=np.eye(3);W[:2,:2]=M
        W[:2,2]=np.array([r['cx'],r['cy']])+shift-M@np.array([128.,128.])
        bounds=[circle_bounds(W@H,1.) for H in regions]
        if all(b is not None for b in bounds):
            lo=np.min([b[:,0] for b in bounds],axis=0)-4;hi=np.max([b[:,1] for b in bounds],axis=0)+4
            period=native79.summarize(q,np.r_[lo,hi],t,1280,720)
        else:period=dict(qualified=False,frequency_hz=None,reason='unbounded_conic')
        value=bank.feed(t,packed,solve_pose=round((t-self.initialized_t)*50)%2==0,current_period=period)
        supported=[np.asarray(v['H']) for v in value['modes'].values()
                   if v['weighted_phase_concentration']>=.5 and v['informative_ring_sectors']>=9
                   and v.get('current_pixel_period_agreement')]
        if supported:
            if track is None:self.regions[i]=supported
            else:track['regions']=supported
        value['rectification_rotor']=dict(r);value['selected_events']=len(ev)
        value['visual_slot_epoch']=self.slot_epoch[i]
        if value['selected']:
            ellipse=image_ellipse(value['selected']['H'],r,shift,1.)
            if ellipse:
                center,shape=ellipse;eig,U=np.linalg.eigh(shape)
                ratio=float(np.sqrt(eig[0]/eig[1]))
                value['conditional_conic_center_image_xy']=center.tolist()
                value['current_projection']=dict(center=center.tolist(),major=float(2*np.sqrt(eig[1])),
                    minor=float(2*np.sqrt(eig[0])),minor_to_major=ratio,
                    normal_axis_unoriented=U[:,0].tolist(),normal_sign_observable=False,
                    side_degenerate=ratio<.15,assumption='weak perspective projected circular disk')
        return value

    def shadow_recovery(self,frame,t,e,xy,shift,q,current):
        output=dict(frame);spawned=[];diagnostics={}
        pending={i:self.pool.submit(self.rotor_step,i,t,e,xy,shift,q,track) for i,track in self.shadows.items()}
        for i,job in pending.items():
            track=self.shadows[i];candidate=job.result()
            if candidate['valid']:track['last_valid_t']=t
            used=False;unique=True
            if candidate['valid'] and not frame[str(i)]['valid']:
                c=np.asarray(candidate['current_projection']['center']);major=candidate['current_projection']['major']
                for j,v in output.items():
                    if int(j)==i or not v['valid']:continue
                    other=v['current_projection'];sep=np.linalg.norm(c-np.asarray(other['center']))
                    if sep<.35*min(major,other['major']):unique=False
                if unique:
                    candidate['observation_source']='current_conic_shadow_recovery'
                    candidate['shadow_created_t']=track['created_t'];output[str(i)]=candidate;used=True
            diagnostics[str(i)]=dict(candidate_valid=candidate['valid'],used=used,unique_current_center=unique,
                primary_valid=frame[str(i)]['valid'],created_t=track['created_t'],
                candidate_frequency_hz=candidate['selected']['phase_hz'] if candidate['valid'] else None)
            if t-track['created_t']>=.5-1e-8 and t-track['last_valid_t']>=.1-1e-8:
                track['bank'].close();del self.shadows[i]
        strong=[i for i in self.banks if frame[str(i)]['valid']]
        if self.current_four_streak>=2 and len(strong)<4 and all(t-self.last_seen[i]<=.5+1e-8 for i in self.banks):
            expected=np.asarray([self.last_good_centers[i]+shift for i in range(4)])
            detected=np.asarray([[r['cx'],r['cy']] for r in current])
            distance=np.linalg.norm(expected[:,None,:]-detected[None,:,:],axis=2)
            ii,jj=linear_sum_assignment(distance);assignment=dict(zip(ii.tolist(),jj.tolist()))
            fit=all(distance[i,assignment[i]]<=.5*current[assignment[i]]['major'] for i in strong)
            if fit:
                for i in self.banks:
                    if i in strong or i in self.shadows or t-self.last_seen[i]<.1-1e-8 or t-self.shadow_last_spawn.get(i,-1e9)<.3:continue
                    j=assignment[i];r=current[j];unique=float(np.delete(distance[i],j).min()-distance[i,j])
                    if distance[i,j]>.75*r['major'] or unique<.25*r['major']:continue
                    a=np.deg2rad(r['angle']);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
                    shape=R@np.diag([(r['major']/2)**2,(r['minor']/2)**2])@R.T
                    margin=np.sqrt(np.diag(shape));center=np.asarray([r['cx'],r['cy']])
                    seed=native79.summarize(q,np.r_[center-margin-4,center+margin+4],t,1280,720)
                    if not seed.get('qualified') or not seed.get('frequency_hz'):continue
                    local=dict(r);local['cx']-=shift[0];local['cy']-=shift[1]
                    self.shadows[i]=dict(bank=SignBank(seed['frequency_hz'],2,self.factory),rotor=local,
                        regions=[np.array([[80.,0,128],[0,80,128],[0,0,1]])],created_t=t,last_valid_t=t)
                    self.shadow_last_spawn[i]=t
                    spawned.append(dict(slot=i,matched_current_conic=j,distance_px=float(distance[i,j]),
                        uniqueness_margin_px=unique,current_seed_hz=seed['frequency_hz'],primary_replaced=False))
        return output,dict(spawned=spawned,candidates=diagnostics)

    def step(self,e,t,body,current,diagnostics,coefficients):
        begin=time.perf_counter();q=native79.checked_process(self.engine,e,self.buffer,1280,720)
        c=coefficients.get('matched_body') or {};velocity=np.asarray(c.get('velocity',[0.,0.]))
        body_period,unused_points=measure_current(q,body,t,velocity,1280,720)
        fallback_mode=('body_periodic_axis' if body_period['axis_valid'] else
                       'current_body_periodicity' if body_period['qualified'] else 'insufficient_rotation_evidence')
        if body.get('detected'):self.last_body_t=t
        if self.last_body_t is not None and t-self.last_body_t>.5:self.reset()
        rotors=current.get('count',[])
        eligible=body.get('detected') and diagnostics.get('count',{}).get('current') and len(rotors)==4
        eligible=eligible and all(r.get('mode')=='ellipse' and not r.get('inferred_shape') for r in rotors)
        self.current_four_streak=self.current_four_streak+1 if eligible else 0
        if self.rotors is None:
            initialized=bool(eligible and self.initialize(rotors,t,body,q))
            return dict(t=t,initialized_now=initialized,valid=False,reason='initialized_waiting_for_new_events' if initialized else 'awaiting_four_current_qualified_conics',
                        epoch=self.epoch,body_periodicity=body_period,observation_mode=fallback_mode,
                        raw_only=True,processing_ms=(time.perf_counter()-begin)*1000)
        if not body.get('detected'):
            return dict(t=t,valid=False,reason='no_current_NN_body',body_periodicity=body_period,
                        observation_mode=fallback_mode,raw_only=True,processing_ms=(time.perf_counter()-begin)*1000)
        shift=np.asarray(body['center'],float)-self.anchor_body
        offset=(e['t']/1e6-t)[:,None]*velocity+shift
        xy=np.c_[e['x'],e['y']]-offset
        jobs=[self.pool.submit(self.rotor_step,i,t,e,xy,shift,q) for i in range(4)]
        frame={str(i):job.result() for i,job in enumerate(jobs)}
        for i in self.banks:
            if frame[str(i)]['valid']:
                self.last_seen[i]=t;self.last_frequency[i]=frame[str(i)]['selected']['phase_hz']
                if 'current_projection' in frame[str(i)]:self.last_good_centers[i]=np.asarray(frame[str(i)]['current_projection']['center'])-shift
        guidance=None;supported=[i for i in self.banks if frame[str(i)]['valid']]
        if len(supported)==4:self.last_complete_t=t
        if (self.allow_current_reanchor and len(supported)<4 and self.current_four_streak>=2
                and self.last_complete_t is not None and t-self.last_complete_t>=.3-1e-8
                and t-self.last_initialization_t>=.5-1e-8):
            old_epoch=self.epoch
            if self.initialize(rotors,t,body,q):
                return dict(t=t,valid=False,reason='fresh_current_conics_reanchor_waiting_for_new_phase',
                    epoch=self.epoch,previous_epoch=old_epoch,reanchored_now=True,initialized_t=t,
                    rotor_slots_are_new_epoch=True,force_valid=False,raw_only=True,
                    processing_ms=(time.perf_counter()-begin)*1000)
        partial_reanchors=[]
        if (self.allow_partial_reanchor and len(supported)<4 and self.current_four_streak>=2
                and all(t-self.last_seen[i]<=.5+1e-8 for i in self.banks)):
            expected=np.asarray([self.last_good_centers[i]+shift for i in range(4)])
            detected=np.asarray([[r['cx'],r['cy']] for r in rotors])
            distance=np.linalg.norm(expected[:,None,:]-detected[None,:,:],axis=2)
            ii,jj=linear_sum_assignment(distance);assignment=dict(zip(ii.tolist(),jj.tolist()))
            strong_fit=all(distance[i,assignment[i]]<=.5*rotors[assignment[i]]['major'] for i in supported)
            if strong_fit:
                for i in self.banks:
                    if i in supported or t-self.last_seen[i]<.1-1e-8 or t-self.last_guided[i]<.3-1e-8:continue
                    j=assignment[i];r=rotors[j];alternatives=np.delete(distance[i],j)
                    unique=float(alternatives.min()-distance[i,j])
                    if distance[i,j]>.75*r['major'] or unique<.25*r['major']:continue
                    a=np.deg2rad(r['angle']);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
                    shape=R@np.diag([(r['major']/2)**2,(r['minor']/2)**2])@R.T
                    margin=np.sqrt(np.diag(shape));center=np.asarray([r['cx'],r['cy']])
                    seed=native79.summarize(q,np.r_[center-margin-4,center+margin+4],t,1280,720)
                    if not seed.get('qualified') or not seed.get('frequency_hz'):continue
                    self.banks[i].close();self.banks[i]=SignBank(seed['frequency_hz'],2,self.factory)
                    r=dict(r);r['cx']-=shift[0];r['cy']-=shift[1];self.rotors[i]=r
                    self.regions[i]=[np.array([[80.,0,128],[0,80,128],[0,0,1]])]
                    self.last_guided[i]=t;self.slot_epoch[i]+=1
                    partial_reanchors.append(dict(slot=i,matched_current_conic=j,distance_px=float(distance[i,j]),
                        uniqueness_margin_px=unique,seed_frequency_hz=seed['frequency_hz'],
                        classified_as_observation=False,new_raw_phase_required=True))
                if partial_reanchors:
                    self.proxy_reference=None;self.proxy_signature=None;self.proxy_epoch+=1
        if len(supported)==3 and not partial_reanchors:
            missing=next(i for i in self.banks if i not in supported)
            if t-self.last_seen[missing]>=.1-1e-8 and t-self.last_guided[missing]>=.3-1e-8:
                conics={i:image_ellipse(frame[str(i)]['selected']['H'],self.rotors[i],shift,1.) for i in supported}
                prior=missing_ellipse(conics,missing,self.opposite) if all(v is not None for v in conics.values()) else None
                if prior:
                    guidance=dict(t=t,missing=missing,ellipse=dict(prior),classified_as_observation=False)
                    self.banks[missing].close();self.banks[missing]=SignBank(self.last_frequency[missing],2,self.factory)
                    prior['cx']-=shift[0];prior['cy']-=shift[1];self.rotors[missing]=prior
                    self.regions[missing]=[np.array([[80.,0,128],[0,80,128],[0,0,1]])]
                    self.last_guided[missing]=t
                    self.slot_epoch[missing]+=1;self.proxy_reference=None;self.proxy_signature=None;self.proxy_epoch+=1
        primary_frame=frame;shadow=None
        if self.allow_shadow_recovery:
            frame,shadow=self.shadow_recovery(primary_frame,t,e,xy,shift,q,rotors)
            supported=[int(i) for i,v in frame.items() if v['valid']]
            signature=tuple((i,v.get('observation_source','primary'),v.get('shadow_created_t')) for i,v in frame.items())
            if signature!=self.output_source_signature:
                self.proxy_reference=None;self.proxy_signature=None;self.proxy_epoch+=1;self.output_source_signature=signature
        proxy=None
        if len(supported)==4:
            signature=tuple(frame[str(i)]['selected']['phase_to_pixel_harmonic_hypothesis'] for i in range(4))
            energy=sum(frame[str(i)]['selected']['phase_hz']**2 for i in range(4))
            if self.proxy_reference is None:self.proxy_reference=energy;self.proxy_signature=signature
            proxy=dict(sum_model_phase_hz_squared=energy,
                relative_to_first_supported=energy/self.proxy_reference if signature==self.proxy_signature else None,
                same_pixel_harmonic_signature=signature==self.proxy_signature,
                mechanical_RPM_or_newtons=False,interpretation='Conditional rotation activity only, not calibrated thrust')
        return dict(t=t,valid=len(supported)==4,reason='all_four_phase_and_current_period_supported' if len(supported)==4 else 'incomplete_current_rotation_evidence',
            initialized_t=self.initialized_t,epoch=self.epoch,rotors=frame,guidance=guidance,body_translation=shift.tolist(),
            partial_reanchors=partial_reanchors,rotation_proxy_epoch=self.proxy_epoch,
            shadow_recovery=shadow,primary_rotors=primary_frame if self.allow_shadow_recovery else None,
            body_periodicity=body_period,
            observation_mode='four_rotor_phase' if len(supported)==4 else ('partial_rotor_phase' if supported else fallback_mode),
            supported_candidates=len(supported),conditional_rotation_proxy=proxy,force_valid=False,raw_only=True,
            processing_ms=(time.perf_counter()-begin)*1000)

    def close(self):
        if self.closed:return
        self.closed=True;self.pool.shutdown(wait=True,cancel_futures=True);self.reset()
