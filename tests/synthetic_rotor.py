"""Controlled rotating-edge event generator; test-only, not flight data."""
import numpy as np

DTYPE=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')])


def rotating_edges(rps=120.,ratio=.4,duration=.16,occlusion=False,velocity=(0,0),angle=.2):
    rng=np.random.default_rng(718)
    times=np.arange(0,duration,50e-6)
    # Two identical blades, each with a brightening and darkening edge. Several
    # radii are sampled, with edge phase offset. No frequency estimator enters.
    t,blade,radius,polarity=np.meshgrid(times,np.arange(2),np.linspace(.55,.98,5),np.arange(2),indexing='ij')
    t=t.ravel();polarity=polarity.ravel()
    phase=2*np.pi*rps*t+blade.ravel()*np.pi+(polarity*2-1)*.045
    x=60*radius.ravel()*np.cos(phase);y=60*ratio*radius.ravel()*np.sin(phase)
    xx=640+x*np.cos(angle)-y*np.sin(angle)+velocity[0]*t
    yy=360+x*np.sin(angle)+y*np.cos(angle)+velocity[1]*t
    keep=(x>0) if occlusion else np.ones(len(t),bool)
    events=np.zeros(keep.sum(),dtype=DTYPE)
    events['x']=np.round(xx[keep]+rng.normal(0,.15,keep.sum())).astype('u2')
    events['y']=np.round(yy[keep]+rng.normal(0,.15,keep.sum())).astype('u2')
    events['p']=polarity[keep];events['t']=np.round(t[keep]*1e6).astype('i8')
    events.sort(order='t')
    rotor=dict(cx=640+velocity[0]*duration,cy=360+velocity[1]*duration,
               major=120.,minor=120*ratio,angle=float(np.rad2deg(angle)),ratio=ratio,
               axis_only=ratio<.08,geometry_quality=.9,activity=1000)
    return events,rotor
