"""Native integration-window ablation and past-only translational compensation."""
from pathlib import Path
import importlib.util
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('reused_stage18_window',ROOT/'research/stage18/frontend18.py')
inc=importlib.util.module_from_spec(spec);spec.loader.exec_module(inc)
RecoveryFrontend=inc.old.RecoveryFrontend;counts=inc.old.counts
METHODS={'native20':(20000,False),'native10':(10000,False),'native5':(5000,False),'warp20':(20000,True)}


def select(events,t,window_us):
    end=round(t*1e6);return events[(events['t']>=end-window_us)&(events['t']<end)]


def warp(events,t,velocity,w,h):
    if not len(events):return events
    age=t-events['t']/1e6
    x=np.rint(events['x'].astype(float)+age*velocity[0]);y=np.rint(events['y'].astype(float)+age*velocity[1])
    keep=(age>=0)&(x>=0)&(x<w)&(y>=0)&(y<h)
    result=events[keep].copy();result['x']=x[keep].astype(events.dtype['x']);result['y']=y[keep].astype(events.dtype['y'])
    return result


class Observer:
    def __init__(self,w,h,method):
        self.w,self.h,self.method=w,h,method;self.geometry=RecoveryFrontend(w,h,'neighbor10')

    def step(self,events,t):
        window,compensate=METHODS[self.method];e=select(events,t,window);used_velocity=np.zeros(2);g=self.geometry
        if compensate and g.last_seen is not None and 0<t-g.last_seen<=.16:
            used_velocity=g.velocity.copy();e=warp(e,t,used_velocity,self.w,self.h)
        prepared=counts(e,self.w,self.h);prepared['on']/=window/10000;prepared['off']/=window/10000
        observation,_=g.detect_prepared(e,t,prepared)
        return dict(observation=observation,candidates=g.candidates,audit=g.audit,window_us=window,
            retained_events=len(e),warp_velocity=used_velocity.tolist())
