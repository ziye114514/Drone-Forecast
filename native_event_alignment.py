"""Exact project-stage21 event translation reused without importing historical frontends.

Original: research/stage21/windows21.py; source SHA256 772f201237418f722b2025ef36a92da276ce9fdb2c99cd757c173bf7ccb38dbd.
Only a known past velocity is allowed; no original event array is edited.
"""
import numpy as np

def warp(events,t,velocity,w,h):
    if not len(events):return events
    age=t-events['t']/1e6
    x=np.rint(events['x'].astype(float)+age*velocity[0]);y=np.rint(events['y'].astype(float)+age*velocity[1])
    keep=(age>=0)&(x>=0)&(x<w)&(y>=0)&(y<h)
    result=events[keep].copy();result['x']=x[keep].astype(events.dtype['x']);result['y']=y[keep].astype(events.dtype['y'])
    return result

def past_warp(events,cutoff_s,velocity,velocity_source_s,window_us=33333,width=1280,height=720):
    begin=cutoff_s-window_us/1e6
    if velocity_source_s is None or begin-velocity_source_s>.16:
        return events,dict(applied=False,reason='missing_or_stale_velocity',source_t=velocity_source_s)
    if velocity_source_s>begin+1e-7:
        raise ValueError('Velocity must predate the complete discovery+validation window')
    velocity=np.asarray(velocity,float)
    if velocity.shape!=(2,) or not np.isfinite(velocity).all():raise ValueError('Finite2D velocity required')
    result=warp(events,cutoff_s,velocity,width,height)
    return result,dict(applied=bool(np.any(velocity)),source_t=velocity_source_s,
        window_start_s=begin,velocity_px_s=velocity.tolist(),max_window_displacement_px=(velocity*window_us/1e6).tolist(),
        timestamps_unchanged=True,original_source='project_stage21_exact_warp')
