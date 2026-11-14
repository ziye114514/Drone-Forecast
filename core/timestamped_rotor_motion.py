"""Translation-only temporal rectification for the MOTION image.

Geometry, period estimation, masks used to discover a target, and the RAW
coordinates stay unchanged. Use only the previously observed image velocity
to map selected events to the source window midpoint. Bilinear deposition
avoids integer-quantized motion. No events after the cutoff are available.
"""
import numpy as np


def render(events,t,mask,velocity,window_us=33333):
    end=round(t*1e6);h,w=mask.shape
    keep=(events['t']>=end-window_us)&(events['t']<end)
    e=events[keep&mask[events['y'],events['x']]]
    v=np.asarray(velocity if velocity is not None else [0.,0.],float)
    dt=(end-window_us*.5-e['t'].astype(float))*1e-6
    x=e['x'].astype(float)+v[0]*dt;y=e['y'].astype(float)+v[1]*dt
    ix=np.floor(x).astype(np.int64);iy=np.floor(y).astype(np.int64)
    fx=x-ix;fy=y-iy;planes=[];mass=0.
    for polarity in (True,False):
        value=np.zeros(h*w,float);p=(e['p']>0)==polarity
        for ox,oy,weights in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)):
            xx=ix+ox;yy=iy+oy;valid=p&(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
            value+=np.bincount(yy[valid]*w+xx[valid],weights=weights[valid],minlength=h*w)
        mass+=float(value.sum());planes.append(value.reshape(h,w).astype(np.float32))
    support=(planes[0]+planes[1])>0
    return tuple(planes),support,dict(selected_original_events=len(e),deposited_mass=mass,
        velocity_from_previous_native_displacement=v.tolist(),reference='source_window_midpoint',
        original_geometry_and_period_coordinates_unchanged=True,mechanical_RPM_created=False)
