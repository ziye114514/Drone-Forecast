"""Local held-out phase evidence, retaining each pixel's template phase.

Sums projection evidence, never raw opposite-polarity signals across pixels.
All eligible pixels enter before validation; regions are not motor identities.
"""
import numpy as np


def local_pixel_phase(frame,center,cell_px=64):
    statistics=frame.get('first_harmonic_statistics')
    empty=dict(qualified=False,regions=[],frequency_hz=None,metric_RPM_measured=False,
        rotor_count_inferred=False,independent_sign_noise_assumption=True,
        common_periodic_light_can_pass=True)
    if statistics is None:return dict(empty,reason='native_first_harmonic_evidence_unavailable')
    pixels=np.asarray(frame['period_pixels_unique']);f=np.asarray(frame['period_candidate_frequencies'])
    if not len(pixels):return dict(empty,reason='no_eligible_native_pixels')
    xy=np.c_[pixels%1280,pixels//1280]
    grid=np.floor((xy-np.asarray(center))/cell_px).astype(np.int64)
    keys=np.c_[np.repeat(grid,2,axis=0),f.reshape(-1)]
    unique,ids=np.unique(keys,axis=0,return_inverse=True)
    counts=np.bincount(ids);projection=np.bincount(ids,weights=statistics[:,[0,2]].reshape(-1))
    energy=np.bincount(ids,weights=statistics[:,[1,3]].reshape(-1))
    score=np.where(projection>0,projection**2/(2*np.maximum(energy,1e-12)),0.)
    bound=np.minimum(1.,2*len(unique)*np.exp(-score))
    regions=[];cells={}
    for i in np.flatnonzero((counts>=3)&(bound<=.01)):
        cell=tuple(unique[i,:2].astype(int));record=dict(cell=list(cell),frequency_hz=float(unique[i,2]),
            eligible_pixels=int(counts[i]),first_harmonic_score=float(score[i]),conditional_bound=float(bound[i]))
        cells.setdefault(cell,[]).append(record)
    for records in cells.values():
        # Prefer the lowest independently supported first harmonic. Multiple
        # supported explanations remain explicit; this is not blade-count truth.
        chosen=min(records,key=lambda r:r['frequency_hz'])
        chosen=dict(chosen,other_supported_frequencies=[r['frequency_hz'] for r in records if r is not chosen],
            harmonic_ambiguity_unresolved=len(records)>1)
        regions.append(chosen)
    support={}
    for r in regions:
        entry=support.setdefault(r['frequency_hz'],[0,0.]);entry[0]+=1;entry[1]+=r['first_harmonic_score']
    best=min(support,key=lambda f:(-support[f][0],-support[f][1],f)) if support else None
    frequency=float(best) if best is not None and support[best][0]>=2 else None
    return dict(empty,qualified=frequency is not None,frequency_hz=frequency,regions=regions,
        reason='current_local_pixel_templates' if frequency is not None else 'insufficient_validated_local_regions',
        eligible_pixels=len(pixels),tested_region_frequency_groups=len(unique),cell_px=cell_px,
        competing_frequency_support={str(f):v for f,v in support.items()},
        selection_does_not_weight_by_changing_pixel_density=True,shared_plane_required=False)
