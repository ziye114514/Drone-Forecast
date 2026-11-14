"""Regional timing evidence can recover current support, never motor RPM."""
import cv2
import numpy as np
from regional_timing_period import measure
from moving_event_proposal import activity_roi

def recover(prepared,events,t):
    on,off,meta,dense,local,period=prepared
    native,diagnostic=meta['native_precomputed'];collective=diagnostic.get('regional_seed_groups',0)>0 and diagnostic.get('regional_seed_pixels',0)>=3
    if np.count_nonzero(native)>=12 and (period['accepted_pixels']>=12 or collective):
        return prepared,dict(used=False,reason='native_period_support_already_available')
    roi=activity_roi(on,off)
    if roi is None:return prepared,dict(used=False,reason='no_coherent_activity_proposal')
    cells,info,groups=measure(events,t,candidate_roi=roi)
    info.update(used=False,reason='no_current_regional_timing',regional_groups=groups)
    if not groups:return prepared,info
    count=on+off
    occupancy=np.float32(count>=2)
    neighbors=cv2.boxFilter(occupancy,-1,(3,3),normalize=False,borderType=cv2.BORDER_CONSTANT)
    support=np.uint8((occupancy>0)&(neighbors>=3))
    number,labels=cv2.connectedComponents(support,connectivity=8)
    seeds=cells&(support>0);touched=np.unique(labels[seeds]);touched=touched[touched>0]
    lookup=np.zeros(number,bool);lookup[touched]=True;recovered=lookup[labels]&(count>0)
    if recovered.sum()<12:return prepared,dict(info,reason='periodic_cells_without_coherent_native_support')
    newmeta=dict(meta);newdiag=dict(diagnostic,native_pixels=int(recovered.sum()),regional_seed_groups=len(groups),
        regional_seed_pixels=int(seeds.sum()),regional_timing_support=True,
        spatial_pixel_timing_periods_not_individual_RPM=True)
    newmeta['native_precomputed']=(recovered,newdiag)
    # A regional comb can establish rotor-like support without establishing
    # any individual motor frequency. Revoke unrelated sparse per-pixel
    # hotspots, including local-phase inputs, on regional-only recovery.
    period=dict(period,mask=np.zeros_like(period['mask']),frequency_hz=np.zeros_like(period['frequency_hz']),
        evidence=np.zeros_like(period['evidence']),pixels=np.empty(0,np.int32),
        candidate_frequency_hz=np.empty((0,2),np.float32),eligible_pixels=0,accepted_pixels=0,
        first_harmonic_statistics=np.empty((0,4),np.float32))
    newmeta.update(accepted_pixels=0,eligible_pixels=0,regional_only_motor_frequency_unobserved=True)
    info.update(used=True,reason='current_native_pixels_connected_to_validated_regional_timing',support_pixels=int(recovered.sum()),
        physical_rims_are_not_cell_borders=True,hidden_RPM_invented=False,known_motor_input_revoked=True)
    return (on,off,newmeta,dense,local,period),info
