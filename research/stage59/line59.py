"""Side-view observation pilot; reuse frozen line geometry, never claim rotor ID."""
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.geometry import _line

PARAMS=dict(threshold_multipliers=[1.,1.5,2.],cap_multiplier=2.,min_activity=30.,
            endpoint_quantiles=[.02,.98],line_max_ratio=.14,
            interpretation='Line support is a candidate, not an independently verified rotor or hub. No new production filter or forecasting state.')

def measure(points,max_major):
    p=np.asarray(points,float).reshape(-1,2)
    if len(p)<6:return None
    m=_line(p,max_major)
    if m is None:return None
    a=np.deg2rad(m['angle']);u=np.array([np.cos(a),np.sin(a)])
    q=(p-np.asarray(m['center']))@u
    lo,hi=np.quantile(q,PARAMS['endpoint_quantiles'])
    robust=np.asarray(m['center'])+u*(lo+hi)/2
    end=np.asarray(m['center'])[None,:]+np.array([q.min(),q.max()])[:,None]*u
    return dict(center=np.asarray(m['center']).tolist(),robust_center=robust.tolist(),
                endpoints=end.tolist(),major=m['major'],minor=m['minor'],angle_deg=m['angle'],
                ratio=m['ratio'],residual_px=m['fit_residual_px'],axis_only=True,
                verified_rotor_identity=False,verified_hub=False)

def candidates(density,threshold,roi,max_major):
    mask=np.uint8(density>=threshold)*255
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    found=[]
    for contour in contours:
        m=measure(contour,max_major)
        if m is None:continue
        x,y,w,h=cv2.boundingRect(contour)
        m['activity']=float(density[y:y+h,x:x+w].sum())
        if m['activity']<PARAMS['min_activity']:continue
        if roi is not None and not (roi[0]<=m['center'][0]<roi[2] and roi[1]<=m['center'][1]<roi[3]):continue
        m['bbox']=[x,y,x+w,y+h];found.append(m)
    return sorted(found,key=lambda r:r['activity'],reverse=True),mask

def axis_weight(ratio,ellipse_support,line_support):
    """Illustrative bounded gate, not a calibrated likelihood or production rule.

    Uses undirected-axis double angles elsewhere; never average opposite normals.
    Both experts must describe the same tracked object before this is meaningful.
    """
    side=float(np.clip((.30-ratio)/(.30-.08),0,1))
    a=side*max(0.,line_support);b=(1-side)*max(0.,ellipse_support)
    return a/(a+b) if a+b>0 else None
