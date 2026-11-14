"""Reuse geometry._line directly on native ON/OFF pixels in the body ROI.

This bypasses rotor segmentation/identity. An axis is not a body centre, a
signed3D normal, or proof of exactly two visible physical motors.
"""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.geometry import _line


def measure(frame,body,prepared=None):
    out=dict(valid=False,source='whole_body_native_ONOFF',sign_observed=False,
        physical_motor_identity_resolved=False,center_is_body_measurement=False)
    if not body.get('detected') or body.get('confidence',0.)<.45:return dict(out,reason='no_strong_body_roi')
    box=np.asarray(body.get('bbox',[]),float)
    if box.shape!=(4,) or not np.isfinite(box).all():return dict(out,reason='invalid_roi')
    if prepared is not None:
        from research.stage6.silhouette import body_measurement
        # The existing shared pair is normalized to10ms. Restore native20ms
        # min(ON,OFF) density; this is event count, not reconstructed intensity.
        density=np.minimum(prepared['on'],prepared['off'])*2.
        m=body_measurement(density,box)
        if not m.get('valid') or m.get('confidence',0.)<.45:
            return dict(out,source='whole_body_ONOFF_count_density',reason=m.get('reason'),density_axis=m)
        normal=np.asarray(m['normal_axis_xy'])
        return dict(out,valid=True,source='whole_body_ONOFF_count_density',reason='existing_stage6_current_density_axis',
            center=m['center'],normal_candidates=[normal.tolist(),(-normal).tolist()],
            tangent=[float(np.cos(m['angle_rad'])),float(np.sin(m['angle_rad']))],
            span_px=m['extent_px'],ratio=m['ratio'],density_axis=m,
            no_grayscale_reconstruction=True,parameters_reused='stage6 multithreshold ON/OFF density + PCA/LSD unchanged')
    lo=np.maximum(np.floor(box[:2]).astype(int),0);hi=np.minimum(np.ceil(box[2:]).astype(int),[1279,719])
    if min(hi-lo)<12:return dict(out,reason='small_roi')
    crop=frame[lo[1]:hi[1]+1,lo[0]:hi[0]+1,0]
    measurements=[]
    for name,mask in [('ON',crop==255),('OFF',crop==200),('union',crop!=52)]:
        y,x=np.nonzero(mask);out[name+'_pixels']=len(x)
        if len(x)<70:return dict(out,reason='insufficient_two_polarity_support')
        points=np.c_[x,y].astype(float)+lo
        if len(points)>5000:points=points[::int(np.ceil(len(points)/5000))]
        line=_line(points,max_major=1.5*max(hi-lo))
        if line is None:return dict(out,reason='native_support_not_thin')
        measurements.append(line)
    angles=np.deg2rad([m['angle'] for m in measurements])
    spread=max(abs((a-angles[-1]+np.pi/2)%np.pi-np.pi/2) for a in angles)
    if np.rad2deg(spread)>15:return dict(out,reason='ON_OFF_axes_disagree')
    line=measurements[-1]
    if line['major']<.5*max(hi-lo):return dict(out,reason='support_covers_less_than_half_body_extent')
    normal=np.asarray(line['normal'])
    return dict(out,valid=True,reason='current_two_polarity_whole_body_axis',
        center=line['center'].tolist(),normal_candidates=[normal.tolist(),(-normal).tolist()],
        tangent=[float(np.cos(angles[-1])),float(np.sin(angles[-1]))],
        span_px=line['major'],width_px=line['minor'],ratio=line['ratio'],
        ON_OFF_spread_deg=float(np.rad2deg(spread)),
        parameters_reused='geometry._line width/extent .14; stage6 support70; axis agreement15deg',
        no_grayscale_reconstruction=True)
