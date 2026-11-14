"""Two-half spatial rotation evidence from native temporal Fourier moments.

Reuses the existing two-blade,9-sector,.5 coherence and3:1 directional gates.
These are descriptive image-rotation tests, not calibrated RPM/force truth.
No raw event buffer or160ms CPU time lattice is needed after current geometry.
"""
import numpy as np


def spatial_statistics(pixels,frequency,coefficients,rotors):
    pixels=np.asarray(pixels);frequency=np.asarray(frequency);c=np.asarray(coefficients)
    xy=np.c_[pixels%1280,pixels//1280];records=[]
    for j,r in enumerate(rotors):
        entry=dict(region_index=j,qualified=False,frequency_hz=None,mechanical_RPM_measured=False,
            blade_count_assumed=2,shared_plane_required=False,harmonic_identity_unverified=True)
        if r.get('fov_clipped') or r.get('ratio',0)<=.16:
            records.append(dict(entry,reason='flat_or_clipped_conic'));continue
        a,b=r['major']/2,r['minor']/2;angle=np.deg2rad(r['angle']);delta=xy-[r['cx'],r['cy']]
        x=(delta[:,0]*np.cos(angle)+delta[:,1]*np.sin(angle))/max(a,1e-9)
        y=(-delta[:,0]*np.sin(angle)+delta[:,1]*np.cos(angle))/max(b,1e-9)
        radius=x*x+y*y;inside=(radius>=.25)&(radius<=1.21)&(frequency>0)
        groups=[]
        for f in np.unique(frequency[inside]):
            keep=inside&(frequency==f);theta=np.arctan2(y[keep],x[keep]);z=c[keep]
            sectors=len(np.unique(np.floor((theta+np.pi)*12/(2*np.pi)).astype(int)%12))
            if len(theta)<12 or sectors<9:continue
            z=np.c_[z[:,0]+1j*z[:,1],z[:,2]+1j*z[:,3]]
            plus=np.sum(z*np.exp(2j*theta[:,None]),axis=0)
            minus=np.sum(z*np.exp(-2j*theta[:,None]),axis=0)
            direction=1 if abs(plus[0])>=abs(minus[0]) else -1
            main,mirror=(plus,minus) if direction==1 else (minus,plus)
            concentration=abs(main)/np.maximum(np.sum(abs(z),axis=0),1e-12)
            contrast=(abs(main)**2-abs(mirror)**2)/np.maximum(abs(main)**2+abs(mirror)**2,1e-12)
            drift=float(abs(np.angle(main[1]*np.conj(main[0]))))
            good=bool(np.min(concentration)>=.5 and np.min(contrast)>=.5 and drift<=np.pi/2)
            groups.append(dict(frequency_hz=float(f),qualified=good,sectors=sectors,pixels=len(theta),
                concentration=concentration.tolist(),direction_contrast=contrast.tolist(),
                independent_half_phase_drift_rad=drift,signed_rotation=direction))
        good=[g for g in groups if g['qualified']]
        best=max(good,key=lambda g:g['pixels']) if good else None
        records.append(dict(entry,qualified=best is not None,frequency_hz=None if best is None else best['frequency_hz'],
            reason='native_spatial_rotation_in_both_halves' if best else 'insufficient_current_spatial_phase',groups=groups))
    return dict(rotors=records,qualified_count=sum(r['qualified'] for r in records),
        native_timestamps_preserved=True,per_pixel_sign_mean_removed_per_half=True,
        current_geometry_is_not_independent_pose_truth=True,new_hidden_frequencies=0,
        frequency_hypothesis_was_selected_by_temporal_validation=True,
        statistical_false_alarm_bound_claimed=False)
