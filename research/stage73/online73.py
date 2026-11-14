"""Inline72 force innovations, with unchanged70 guard logic and cheap exact caches."""
from pathlib import Path
from functools import lru_cache
from types import FunctionType
import inspect,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage71'),str(ROOT/'research/stage72'),str(ROOT/'research/stage70')]
from online71 import OnlineExtras as OriginalExtras
from increment72 import IncrementForecast,ADDED as DELTAS
import activity68
from compensation64 import warp
import guard70

@lru_cache(maxsize=256)
def carrier_for(f,n):return np.exp(-2j*np.pi*f*(np.arange(n)+.5)/10000.)
_source=inspect.getsource(activity68.tone).replace('carrier=np.exp(-2j*np.pi*f*(np.arange(a.shape[-1])+.5)/10000.)','carrier=carrier_for(f,a.shape[-1])')
assert 'carrier=carrier_for' in _source
_tone_globals=dict(activity68.tone.__globals__,carrier_for=carrier_for)
exec(compile(_source,'<73 exact carrier memoisation>','exec'),_tone_globals)
_step_globals=dict(activity68.Activity.step.__globals__,tone=_tone_globals['tone'])
class FastActivity(activity68.Activity):step=FunctionType(activity68.Activity.step.__code__,_step_globals)

class FastSupports:
    """Exact64 aligned_total only; unused three support maps are not computed."""
    def __init__(self,w,h):self.w=w;self.h=h;self.density=dict(aligned_total=np.zeros((h,w),np.float32))
    def step(self,events,t,velocity):
        velocity=np.asarray(velocity,float);nonzero=bool(np.any(velocity!=0))
        e=warp(events,t,velocity,self.w,self.h) if nonzero else events
        pix=e['y'].astype(np.int32)*self.w+e['x']
        value=np.bincount(pix,minlength=self.w*self.h).reshape(self.h,self.w).astype(np.float32)
        old=self.density['aligned_total']
        if nonzero:old=cv2.warpAffine(old,np.c_[np.eye(2),velocity*.01].astype(np.float32),(self.w,self.h),flags=cv2.INTER_LINEAR)
        old*=np.exp(-.01/.02);old+=value;self.density['aligned_total']=old

_guard_globals=dict(guard70.Guard.step.__globals__,ADDED=DELTAS)
class IncrementGuard(guard70.Guard):step=FunctionType(guard70.Guard.step.__code__,_guard_globals)

class OnlineExtras(OriginalExtras):
    def __init__(self,w,h):
        super().__init__(w,h);self.activity=FastActivity(w,h);self.supports=FastSupports(w,h)
        self.increment=IncrementForecast();self.increment_guard=IncrementGuard()
    def forecast(self,q,obs,rotor_quality):
        q=super().forecast(q,obs,rotor_quality)
        q=self.increment.step(q,obs,self.current['body'],self.current['activity'],rotor_quality)
        # Reuse unchanged guard mathematics in an isolated namespace, with
        # the new candidates. Preserve all old26 outputs/diagnostics exactly.
        temp=dict(q,event_force_diagnostic=dict(models=q['increment_diagnostic']['models']))
        guarded=self.increment_guard.step(temp,obs)
        for name in guard70.ADDED70:
            if q['valid']:
                q['coefficients']['increment_'+name]=guarded['coefficients'][name]
                q['predictions']['increment_'+name]=guarded['predictions'][name]
        q['increment_guard_diagnostic']=guarded['force_guard_diagnostic']
        return q

def controls():
    from compensation64 import Supports
    dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')]);rng=np.random.default_rng(7301)
    original=Supports(80,60);fast=FastSupports(80,60)
    for j in range(20):
        e=np.zeros(300,dtype);e['x']=rng.integers(0,80,len(e));e['y']=rng.integers(0,60,len(e));e['p']=rng.integers(0,2,len(e));e['t']=rng.integers(j*10000,(j+1)*10000,len(e))
        v=np.array([170.,-130.]) if j%3 else np.zeros(2)
        original.step(e,(j+1)*.01,v);fast.step(e,(j+1)*.01,v)
        np.testing.assert_array_equal(original.density['aligned_total'],fast.density['aligned_total'])
    for f in [100.,119.3,287.5,523.4]:
        a=rng.poisson(2.,(2,1600));assert activity68.tone(a,f)==_tone_globals['tone'](a,f)
    return dict(aligned_total_pixel_exact64=True,phase_arithmetic_exact68=True,
                original_unused_maps_omitted=True,phase_carriers_memoised_only=True)
