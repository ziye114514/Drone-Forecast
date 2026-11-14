"""Anonymous observed rotor-force contrast; no heading or hidden motor fill.

For parallel thrusts, tau=sum(r_i cross F_i*n). With isotropic in-plane
inertia, (r cross F*n) cross n = -F*r. The image-plane first spatial moment
therefore supplies a projected normal-acceleration PROXY, not Newton-metres.
Missing rotors, unknown propeller coefficients and inertia remain unobserved.
"""
import numpy as np


def contrast(positions,frequencies,*,radius_px,coefficient_scales=None,weight_relative_sigma=.04):
    p=np.asarray(positions,float);f=np.asarray(frequencies,float)
    if p.ndim!=2 or p.shape[1]!=2 or f.shape!=(len(p),) or len(p)<2:raise ValueError('At least two current distinct rotor sources required')
    if not np.isfinite(p).all() or not np.isfinite(f).all() or np.any(f<=0) or not np.isfinite(radius_px) or radius_px<=0:raise ValueError('Finite positions, positive frequencies and radius required')
    c=np.ones(len(p)) if coefficient_scales is None else np.asarray(coefficient_scales,float)
    if c.shape!=f.shape or not np.isfinite(c).all() or np.any(c<0):raise ValueError('Nonnegative individual coefficient priors required')
    w=c*(f/300.)**2/4.
    r=(p-p.mean(0))/radius_px
    moment=-np.sum((w-w.mean())[:,None]*r,axis=0)
    cov=np.einsum('i,ij,ik->jk',np.square(weight_relative_sigma*w),r,r)
    return dict(normal_acceleration_contrast_xy=moment.tolist(),weight_covariance_proxy=cov.tolist(),
        individual_f2_weights=w.tolist(),visible_relative_sum=float(w.sum()),
        centre_reference='unweighted visible-source centroid; not hidden body centre',
        radius_normalization_px=float(radius_px),visible_sources=len(p),
        missing_force_and_torque_owner='unobserved_response',physical_motor_names_required=False,
        propeller_coefficients_measured=False,individual_coefficient_priors=c.tolist(),
        torque_Nm_measured=False,mechanical_RPM_measured=False,
        partial_contrast_is_not_complete_torque=True,
        four_sources_alone_do_not_prove_complete_physical_torque=True)


def from_joint_period(period):
    meta=period.get('meta') or {};sources=[]
    for record in meta.get('records',[]):
        if not record.get('qualified') or not record.get('current_force_admitted'):continue
        f=record.get('proxy_twofold_hz');geometry=record.get('result',{}).get('arms',{}).get('joint_pose',{}).get('inferred_geometry') or {}
        if not f or geometry.get('fov_clipped') or 'major' not in geometry:continue
        p=geometry.get('projected_shaft_xy',[geometry.get('cx'),geometry.get('cy')])
        sources.append((str(record['track_id']),p,float(f),float(geometry['major'])/2))
    if len(sources)<2:return dict(qualified=False,reason='fewer_than_two_current_independent_period_sources',current_sources=len(sources))
    if len({s[0] for s in sources})!=len(sources):return dict(qualified=False,reason='duplicate_current_source')
    result=contrast([s[1] for s in sources],[s[2] for s in sources],radius_px=float(np.median([s[3] for s in sources])))
    result.update(qualified=True,reason='current_observed_spatial_f2_contrast',source_ids=sorted(s[0] for s in sources),
        source_positions=[s[1] for s in sources],source_frequencies_hz=[s[2] for s in sources],
        source_is_joint_late_confirmed=True,source_set_is_not_physical_heading=True)
    return result


def from_slot_period(row,period):
    """Current native slot contributions only; no pooled-frequency replication."""
    if period.get('source')!='current_visible_slot_f2_sum':
        return dict(qualified=False,reason='no_current_spatial_slot_contributions')
    n=np.asarray(row['structure'].get('axis_xy',[]),float)
    if n.shape!=(2,) or n@n<1e-12:return dict(qualified=False,reason='no_current_normal')
    assignments={a['model_id']:a['region_index'] for a in period.get('mapping',{}).get('assignments',[])}
    sources=[]
    for key,value in period.get('observed_components',{}).items():
        if not key.startswith('slot:'):continue
        i=int(key.split(':')[-1]);index=assignments.get(i)
        if index is None:continue
        geometry=row['geometry']['rotors'][index]
        if geometry.get('fov_clipped') or not geometry.get('current_observation',True):continue
        w=float(np.asarray(value)@n/(n@n))
        if w<=0:continue
        sources.append((key,[geometry['cx'],geometry['cy']],np.sqrt(w*360000),geometry['major']/2))
    if len(sources)<2:return dict(qualified=False,reason='fewer_than_two_current_slot_sources')
    result=contrast([s[1] for s in sources],[s[2] for s in sources],radius_px=float(np.median([s[3] for s in sources])))
    result.update(qualified=True,reason='current_native_spatial_f2_contrast',source_ids=sorted(s[0] for s in sources),
        source_positions=[s[1] for s in sources],source_frequencies_hz=[float(s[2]) for s in sources],
        source_is_joint_late_confirmed=False,source_set_is_not_physical_heading=True)
    return result
