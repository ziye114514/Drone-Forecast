"""Optional CPU joint ring-geometry/phase observation using the author core.

This is an observation module, not a drone detector, RPM ground truth or
trajectory predictor. Caller supplies geometry and motion from discovery only.
Native author headers/DLL are external local dependencies, not distributed.
"""
import numpy as np
from research.helix_reference.reference_api import AuthorTracker
from angular_phase_candidates import ring_phase_candidates,ring_phase_samples
from ellipse_visibility import ellipse_fov_clipped

def projected_geometry(ellipse,H):
    a=np.deg2rad(ellipse['angle']);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    R=R@np.diag([ellipse['major']/160,ellipse['minor']/160]);M=np.eye(3)
    M[:2,:2]=R;M[:2,2]=[ellipse['cx'],ellipse['cy']]-R@np.array([128,128])
    J=M@np.asarray(H);inv=np.linalg.inv(J);C=inv.T@np.diag([1.,1.,-1.])@inv
    centre=-np.linalg.solve(C[:2,:2],C[:2,2]);shape=(centre@C[:2,:2]@centre-C[2,2])*np.linalg.inv(C[:2,:2])
    eig,U=np.linalg.eigh(shape)
    if eig.min()<=0 or not np.isfinite(eig).all():raise ValueError('Projected circle is not a finite ellipse')
    r=dict(cx=float(centre[0]),cy=float(centre[1]),major=float(2*np.sqrt(eig[-1])),minor=float(2*np.sqrt(eig[0])),
        angle=float(np.rad2deg(np.arctan2(U[1,-1],U[0,-1]))))
    r.update(fov_clipped=ellipse_fov_clipped(r),projected_shaft_xy=(J[:2,2]/J[2,2]).tolist(),
        is_inferred_geometry=True,position_is_not_independent_ground_truth=True)
    return r

def observe_ring(e,r,t,velocity=None,*,author_dll,geometry_source_end_s,velocity_source_end_s=None,compare_phase_only=False,phase_hint=None,initialization_only=False):
    mid_s=(round(t*1e6)-33333+33333//2)/1e6
    source_limit=t if initialization_only else mid_s
    if geometry_source_end_s is None or geometry_source_end_s>source_limit+1e-9:
        raise ValueError('Initial geometry must predate the validation half')
    if velocity is not None and (velocity_source_end_s is None or velocity_source_end_s>source_limit+1e-9):
        raise ValueError('Translation initialization must predate validation')
    if phase_hint is not None:
        if phase_hint['source_t']>source_limit+1e-9:raise ValueError('Phase hint must predate validation')
        seed=dict(angular_harmonic=int(phase_hint['angular_order']),
            candidate_signed_phase_hz=float(phase_hint['twofold_hz'])*int(phase_hint['angular_order'])/2*int(phase_hint['rotation_sign']),
            initialization='past_confirmed_same_spatial_source_not_current_measurement',
            phase_hint_source_t=phase_hint['source_t'])
    else:seed=ring_phase_candidates(e,r,t,velocity)
    data,reason=ring_phase_samples(e,r,t,velocity)
    if data is None:return dict(control=seed,reason=reason,arms={})
    m=seed['angular_harmonic'];sf=seed['candidate_signed_phase_hz'];sign=1 if sf>0 else -1
    v=np.zeros(2) if velocity is None else np.asarray(velocity);a=np.deg2rad(r['angle']);co,si=np.cos(a),np.sin(a)
    d=np.c_[e['x'],e['y']]-[r['cx'],r['cy']]-(e['t']/1e6-t)[:,None]*v
    u=np.c_[(d[:,0]*co+d[:,1]*si)/(r['major']/2),(-d[:,0]*si+d[:,1]*co)/(r['minor']/2)]
    keep=np.all(abs(u)<=2.,axis=1);native=e[keep];u=u[keep];xy=np.rint(128+80*u).astype(np.int32)
    early=native['t']<data['mid'];late=~early
    if min(early.sum(),late.sum())<32:return dict(control=seed,reason='few_native_samples',arms={})
    train_e=data['events'][data['train']];theta=data['theta'][data['train']];w=data['weight'][data['train']]
    z=np.sum(w*np.exp(1j*(m*theta-2*np.pi*sf*(train_e['t']-data['begin'])/1e6)))
    first=int(native['t'][early][0]);initial_phase=2*np.pi*abs(sf)*(first-data['begin'])/1e6+sign*np.angle(z)
    def score(fitted,mask):
        H=np.asarray(fitted['H']);inv=np.linalg.inv(H)
        point=np.c_[xy[mask],np.ones(mask.sum())]@inv.T
        good=np.isfinite(point).all(1)&(abs(point[:,2])>1e-9)
        normalized=point[good,:2]/point[good,2,None];radius=np.linalg.norm(normalized,axis=1)
        band=(radius>=.2)&(radius<=1.5);normalized=normalized[band];radius=radius[band]
        if len(radius)<32:return dict(qualified=False,reason='few_frozen_pose_events')
        times=native['t'][mask][good][band];weight=np.exp(-.5*((radius-1)/.15)**2)
        angle=np.arctan2(sign*normalized[:,1],normalized[:,0]);opposite=np.arctan2(-sign*normalized[:,1],normalized[:,0])
        last=int(native['t'][early][-1]);tl=(last-first)/1e6
        phase_last=fitted['phi0']+fitted['angular_velocity_rad_s']*tl+.5*fitted['alpha']*tl**2
        age=(times-last)/1e6
        # The paper's instantaneous state at the last discovery event. No
        # validation event updates the author filter or homography.
        phi=phase_last+fitted['angular_velocity_rad_s']*age+.5*fitted['alpha']*age**2-m*fitted['q'][1]
        value=np.sum(weight*np.exp(1j*(phi-m*angle)))/max(weight.sum(),1e-12)
        other=np.sum(weight*np.exp(1j*(phi-m*opposite)))/max(weight.sum(),1e-12)
        sectors=len(np.unique(np.floor((angle[weight>=.1]+np.pi)*12/(2*np.pi)).astype(int)%12))
        return dict(concentration=float(abs(value)),complex_mean=[float(value.real),float(value.imag)],
            opposite_power=float(abs(other)**2),sectors=sectors,events=len(radius))
    arms={}
    for arm in (('phase_only','joint_pose') if compare_phase_only else ('joint_pose',)):
        tracker=AuthorTracker(abs(sf)*60,blades=m,dll_path=author_dll,signed_coordinates=True,
            rotation_sign=sign,initial_phase_rad=initial_phase)
        try:
            ids=np.flatnonzero(early);fitted=None
            for chunk in np.array_split(ids,4):
                if not len(chunk):continue
                fitted=tracker.feed(xy[chunk,0],xy[chunk,1],native['t'][chunk],native['p'][chunk],solve_pose=arm=='joint_pose')
        finally:tracker.close()
        train=score(fitted,early);validation=score(fitted,late)
        qualified=False;agreement=None;contrast=None
        if 'complex_mean' in train and 'complex_mean' in validation:
            z=complex(*train['complex_mean']);zz=complex(*validation['complex_mean'])
            agreement=float(np.real(zz*np.conj(z))/max(abs(z),1e-12))
            power=abs(zz)**2;other=validation['opposite_power'];contrast=float((power-other)/max(power+other,1e-12))
            qualified=bool(abs(z)>=.5 and abs(zz)>=.5 and agreement>=.5 and contrast>=.5 and validation['sectors']>=9)
        try:geometry=projected_geometry(r,fitted['H'])
        except (ValueError,np.linalg.LinAlgError):geometry=dict(invalid=True,fov_clipped=True)
        qualified=qualified and not geometry['fov_clipped']
        arms[arm]=dict(qualified=qualified and not initialization_only,train=train,validation=validation,phase_agreement=agreement,
            initialization_window_score_is_not_an_independent_measurement=initialization_only,
            inferred_geometry=geometry,
            direction_contrast=contrast,author_state=fitted,discovery_state_time_s=int(native['t'][early][-1])/1e6,
            heldout_events_fed_to_author=False)
    return dict(control=seed,arms=arms,initialization_uses_only_discovery=True,blades_are_selected_angular_order_not_GT=True,
        geometry_source_end_s=geometry_source_end_s,velocity_source_end_s=velocity_source_end_s,force_input_enabled=False,
        initialization_only=initialization_only,current_frequency_measurement_admissible=not initialization_only)

