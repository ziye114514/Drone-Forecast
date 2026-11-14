"""Current image observations between fresh rotor-period confirmations.

Object identity expires after0.12s without a current period confirmation.
Within that time, previous support only seeds a CURRENT bipolar component.
Bidirectional matching, original extent guard, and real current event support
are required. A retained template is never itself a position observation.
No current motor frequency/thrust measurement is created by this bridge.
"""
import numpy as np
import cv2
from curvature_visibility import seeded_outline
from rotor_support_front import estimate_region_translation
from native_cycle_consistency import qualify


def bounded_seed(mask,dx,dy):
    """A past rotor leaving one sensor edge must not reappear at the other."""
    h,w=mask.shape;dx=int(round(dx));dy=int(round(dy));out=np.zeros_like(mask)
    x=max(0,-dx);X=min(w,w-dx);y=max(0,-dy);Y=min(h,h-dy)
    if x<X and y<Y:out[y+dy:Y+dy,x+dx:X+dx]=mask[y:Y,x:X]
    return out


def recover(raw,front,t,last_period_t,include_displacement=False):
    info=dict(used=False,current_period_measured=False,current_frequency_created=False,
        known_thrust_added=False,position_requires_current_image_match=True)
    if last_period_t is None or not 0<t-last_period_t<=.12+1e-8:
        return raw,dict(info,reason='rotor_identity_period_confirmation_expired')
    if front.previous is None or front.previous_mask is None or front.last_t is None or front.last_velocity is None:
        return raw,dict(info,reason='no_past_confirmed_motion_template')
    dt=t-front.last_t
    if not 0<dt<=.12:return raw,dict(info,reason='motion_template_expired')
    prior=front.last_velocity*dt;seed=bounded_seed(front.previous_mask,*prior)
    mask=seeded_outline(raw[0],raw[1],seed)
    if mask.sum()<12:return raw,dict(info,reason='no_current_bipolar_outline')
    current_box=cv2.boundingRect(mask.view(np.uint8));old_box=cv2.boundingRect(front.previous_mask.view(np.uint8))
    if max(current_box[2:])<.5*max(old_box[2:]):
        return raw,dict(info,reason='current_support_is_a_small_fragment')
    radius=int(np.ceil(40*dt*30));prediction=prior
    region_delta=np.array(current_box[:2])+np.array(current_box[2:])/2-np.array(old_box[:2])-np.array(old_box[2:])/2
    if np.max(abs(region_delta-prior))>radius and np.max(abs(region_delta))<=160:prediction=region_delta
    options=dict(native_matching=front.fast_geometry,subpixel=front.subpixel_motion,max_shift=radius,
        coarse_matching=front.coarse_motion,interior_border=front.border_motion,centered_matching=front.centered_motion)
    current=(mask.astype(np.float32),)*2 if getattr(front,'occupancy_motion',False) else (raw[0],raw[1])
    forward=estimate_region_translation(front.previous,front.previous_mask,current,current_support=mask,predicted_shift=prediction,**options)
    reverse=estimate_region_translation(current,mask,front.previous,current_support=front.previous_mask,predicted_shift=-np.asarray(prediction),**options)
    if forward is None:return raw,dict(info,reason='no_current_forward_match')
    checked=qualify(forward,reverse)
    if not checked.get('current_displacement_qualified'):return raw,dict(info,reason='current_match_cycle_rejected',cycle=checked.get('cycle_consistency'))
    meta=dict(raw[2]);diag=dict(meta['native_precomputed'][1])
    diag.update(native_pixels=int(mask.sum()),mask_support_pixels=int(mask.sum()),
        prior_rotor_identity_seed_not_new_current_period=True,real_current_bipolar_pixels=int(mask.sum()))
    meta.update(native_precomputed=(mask,diag),regional_only_motor_frequency_unobserved=True,
        current_spatial_track_qualified=True,
        current_spatial_track=dict(reason='current_image_bridge_of_recently_period_confirmed_rotor',
            current_period_measured=False,period_confirmation_age_s=t-last_period_t))
    if include_displacement:
        meta['verified_native_displacement']=dict(source_t=front.last_t,cutoff_t=t,dt_s=dt,estimate=checked)
    info.update(used=True,reason='current_bipolar_outline_and_bidirectional_match',
        period_confirmation_age_s=t-last_period_t,maximum_identity_hold_s=.12,
        original_support_pixels=int(mask.sum()),cycle=checked['cycle_consistency'])
    return (raw[0],raw[1],meta,raw[3],raw[4],raw[5]),info
