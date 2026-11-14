"""One discovery-selected angular/temporal harmonic, frozen late validation.

Generalizes this project's observed_ring_phase (default twofold) without
claiming that the chosen angular order identifies the physical blade count.
All native timestamps retained. Geometry/translation must come from discovery.
"""
import numpy as np
from predictive_harmonics import BASE_FREQUENCIES
from angular_projection import project as factorized_project

def ring_phase_samples(events,ellipse,t,velocity=None):
    end=round(t*1e6);begin=end-33333;mid=begin+33333//2
    e=events[(events['t']>=begin)&(events['t']<end)]
    angle=np.deg2rad(ellipse['angle']);axes=np.array([ellipse['major'],ellipse['minor']])/2
    extent=np.sqrt(np.array([[np.cos(angle)**2,np.sin(angle)**2],[np.sin(angle)**2,np.cos(angle)**2]])@(axes**2))
    center=np.array([ellipse['cx'],ellipse['cy']])
    clipped=bool(np.any(center-extent<0) or np.any(center+extent>=[1280,720]))
    if clipped or ellipse.get('fov_clipped') or ellipse['minor']/ellipse['major']<=.16:
        return None,'clipped_or_flat_ring'
    v=np.zeros(2) if velocity is None else np.asarray(velocity)
    d=np.c_[e['x'],e['y']]-[ellipse['cx'],ellipse['cy']]-(e['t']/1e6-t)[:,None]*v
    a=np.deg2rad(ellipse['angle']);co,si=np.cos(a),np.sin(a)
    x=(d[:,0]*co+d[:,1]*si)/(ellipse['major']/2);y=(-d[:,0]*si+d[:,1]*co)/(ellipse['minor']/2)
    radius=np.hypot(x,y);keep=(radius>=.2)&(radius<=1.5)
    e=e[keep];radius=radius[keep];theta=np.arctan2(y[keep],x[keep])
    weight=np.exp(-.5*((radius-1)/.15)**2);train=e['t']<mid;late=~train
    if min(train.sum(),late.sum())<32:return None,'sparse_ring_halves'
    return dict(events=e,theta=theta,weight=weight,train=train,late=late,begin=begin,mid=mid,end=end),None

def ring_phase_candidates(events,ellipse,t,velocity=None):
    empty=dict(qualified=False,phase_hz=None,angular_harmonic=None,mechanical_RPM_measured=False,
        angular_order_is_not_verified_blade_count=True,geometry_must_be_discovery_only=True,
        nominal_camera_false_alarm_not_calibrated=True,all_selected_native_timestamps_retained=True)
    data,reason=ring_phase_samples(events,ellipse,t,velocity)
    if data is None:return dict(empty,reason=reason)
    e,theta,weight,train,late,begin=[data[k] for k in ('events','theta','weight','train','late','begin')]
    positive=np.unique(BASE_FREQUENCIES[:,None]*np.arange(1,5)[None])
    frequencies=np.r_[positive,-positive];f=np.tile(frequencies,4);order=np.repeat(np.arange(1,5),len(frequencies))
    def project(mask,frequencies,orders):
        tt=(e['t'][mask]-begin)/1e6;ww=weight[mask];angle=theta[mask]
        return factorized_project(tt,angle,ww,frequencies,orders)
    z=project(train,f,order);k=int(np.argmax(abs(z)));chosen=float(f[k]);angular=int(order[k])
    left=np.flatnonzero((f==chosen-15)&(order==angular));right=np.flatnonzero((f==chosen+15)&(order==angular))
    if len(left) and len(right):
        power=np.log(np.maximum(abs(z[[left[0],k,right[0]]])**2,1e-12));den=power[0]-2*power[1]+power[2]
        if den < -1e-12:chosen+=15*float(np.clip(.5*(power[0]-power[2])/den,-.5,.5))
    frozen=np.array([chosen,-chosen]);orders=np.array([angular,angular])
    early=project(train,frozen,orders);validation=project(late,frozen,orders)
    power=abs(validation)**2;contrast=float((power[0]-power[1])/max(power.sum(),1e-12))
    agreement=float(np.real(validation[0]*np.conj(early[0]))/max(abs(early[0]),1e-12))
    sectors=len(np.unique(np.floor((theta[late&(weight>=.1)]+np.pi)*12/(2*np.pi)).astype(int)%12))
    qualified=bool(abs(early[0])>=.5 and abs(validation[0])>=.5 and contrast>=.5 and agreement>=.5 and sectors>=9)
    return dict(empty,qualified=qualified,phase_hz=abs(chosen) if qualified else None,angular_harmonic=angular,
        candidate_signed_phase_hz=chosen,conditional_rotation_hz=abs(chosen)/angular if qualified else None,
        discovery_concentration=float(abs(early[0])),validation_concentration=float(abs(validation[0])),
        validation_direction_power_contrast=contrast,phase_agreement=agreement,validation_sectors=sectors,
        train_events=int(train.sum()),validation_events=int(late.sum()),discovery_candidates=len(f),
        frequency_order_direction_frozen_before_validation=True,
        reason='current_ring_phase_validated' if qualified else 'ring_phase_not_validated')
