"""Recover current rotor support without changing geometry/motion coordinates.

Only failed original entry may use a motion-normalized periodic proof. Its
support is backprojected event-by-event to ORIGINAL x/y. Frequency estimates
at virtual aligned pixels are not advertised as native-pixel motor RPM; the
force observer owns all such uncalibrated thrust in its unknown term.
"""
import numpy as np


def qualified(packet):
    native,diag=packet[2]['native_precomputed'];accepted=int(packet[-1]['accepted_pixels'])
    regional=bool(diag.get('regional_seed_groups',0)>0 and diag.get('regional_seed_pixels',0)>=3)
    return bool(native.sum()>=12 and (accepted>=12 or regional))


def backproject(raw,aligned,events,t,velocity):
    if qualified(raw):return raw,dict(used=False,reason='original_period_entry_already_qualified')
    if not qualified(aligned):return raw,dict(used=False,reason='normalized_period_entry_not_qualified')
    end=round(t*1e6);e=events[(events['t']>=end-33333)&(events['t']<end)]
    v=np.asarray(velocity,float)
    if v.shape!=(2,) or not np.isfinite(v).all():raise ValueError('Finite2D alignment velocity required')
    age=t-e['t']/1e6;x=np.rint(e['x'].astype(float)+age*v[0]).astype(int);y=np.rint(e['y'].astype(float)+age*v[1]).astype(int)
    valid=(x>=0)&(x<1280)&(y>=0)&(y<720)
    ids=np.flatnonzero(valid);mask=aligned[2]['native_precomputed'][0];ids=ids[mask[y[ids],x[ids]]]
    native=np.zeros((720,1280),bool);native[e['y'][ids],e['x'][ids]]=True
    if native.sum()<12:return raw,dict(used=False,reason='too_few_original_supported_pixels')
    meta=dict(raw[2]);diag=dict(raw[2]['native_precomputed'][1])
    proof=aligned[2]['native_precomputed'][1]
    seeds=max(int(aligned[-1]['accepted_pixels']),int(proof.get('regional_seed_pixels',0)))
    diag.update(native_pixels=int(native.sum()),mask_support_pixels=int(native.sum()),
        regional_seed_groups=1,regional_seed_pixels=seeds,
        motion_normalized_period_proof=True,original_event_coordinates_retained=True,
        seed_count_is_virtual_aligned_pixels_not_mechanical_rotor_count=True)
    meta.update(native_precomputed=(native,diag),regional_only_motor_frequency_unobserved=True,
        regional_graph_fallback=dict(used=True,reason='motion_tube_period_evidence_on_original_pixels',
            motor_frequency_transported_as_native_measurement=False))
    result=(raw[0],raw[1],meta,raw[3],raw[4],raw[5])
    return result,dict(used=True,reason='original_pixel_tube_supported_by_current_normalized_period',
        original_support_pixels=int(native.sum()),supporting_original_events=len(ids),
        virtual_period_seed_pixels=seeds,known_thrust_added=False,
        hidden_force_owner='lumped_unknown_force',geometry_and_motion_coordinates_unchanged=True)
