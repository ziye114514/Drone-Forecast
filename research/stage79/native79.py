"""Installed Prophesee frequency engine, with audited array-layout adapter.

Pixel flicker frequency is not mechanical RPM. No SDK source/binary is changed
or redistributed. The local Windows binding reports stride16 for24B records;
we use its own native buffer metadata, copy immediately, and test known data.
"""
from collections import deque
import numpy as np
import metavision_sdk_cv as sdk
import metavision_sdk_base as sdk_base

PARAMS=dict(min_hz=60.,max_hz=1200.,window_us=160000,current_us=20000,
    configurations=dict(default7=dict(filter_length=7,diff_thresh_us=1500),
                        strict7=dict(filter_length=7,diff_thresh_us=200),
                        fast3=dict(filter_length=3,diff_thresh_us=200)),
    log_frequency_bin_ratio=1.02,minimum_pixels=16,minimum_mode_fraction=.35,
    maximum_ring_density_ratio=.5,minimum_body_confidence=.45,
    interpretation='Brightness flicker candidates only. Unique pixel votes, '
        'current20ms support, ring common-mode rejection. Fixed design choices '
        'before real data; no position-label tuning or mechanical RPM claim.')

def engine(w,h,name):
    p=PARAMS['configurations'][name]
    return sdk.FrequencyAlgorithm(w,h,p['filter_length'],60.,1200.,p['diff_thresh_us'],False)

def copy_frequency(buffer):
    """Read only the allocated extent certified by the native buffer metadata."""
    q=buffer.numpy();info=buffer._buffer_info()
    assert info.ndim==1 and tuple(info.shape)==q.shape and info.size==len(q)
    assert info.itemsize==q.dtype.itemsize and tuple(info.strides)==(info.itemsize,)
    assert q.dtype.names==('x','y','t','frequency')
    if len(q)==0:return q.copy()
    if q.strides!=tuple(info.strides):
        # q is a borrowed view of the same native allocation. Native metadata
        # bounds shape/strides; never reinterpret the wrongly copied q array.
        assert q.base is not None and q.strides==(16,) and info.itemsize==24
        q=np.lib.stride_tricks.as_strided(q,shape=tuple(info.shape),
                                        strides=tuple(info.strides),writeable=False)
    return q.copy()

def checked_process(algo,e,output,w,h):
    algo.process_events(e,output);q=copy_frequency(output)
    if len(q):
        assert (q['x']<w).all() and (q['y']<h).all()
        assert np.isfinite(q['frequency']).all()
        assert (np.diff(q['t'])>=0).all()
        assert e['t'][0]<=q['t'][0]<=q['t'][-1]<=e['t'][-1]
    return q

def summarize(q,bbox,t,w,h,unique_pixels=False):
    q=q[q['t']>=round(t*1e6)-PARAMS['current_us']]
    assert not len(q) or q['t'].max()<round(t*1e6)
    allowed=(q['frequency']>=60.)&(q['frequency']<=1200.)
    excluded=int((~allowed).sum());q=q[allowed]
    if bbox is None:return dict(qualified=False,reason='no_current_body',pixels=0)
    b=np.asarray(bbox,float).copy();b[:2]=np.maximum(b[:2],0);b[2:]=np.minimum(b[2:],[w,h])
    size=b[2:]-b[:2]
    if min(size)<12:return dict(qualified=False,reason='small_ROI',pixels=0)
    area=float(np.prod(size));pad=np.maximum(size*.25,12)
    outer=np.r_[np.maximum(b[:2]-pad,0),np.minimum(b[2:]+pad,[w,h])]
    ring_area=float(np.prod(outer[2:]-outer[:2])-area)
    # One latest frequency per pixel. Event/burst count is not an independent
    # vote: otherwise one hot pixel can dominate the pooled signal.
    if len(q) and not unique_pixels:
        pix=q['y'].astype(np.int64)*w+q['x'];_,ii=np.unique(pix[::-1],return_index=True)
        q=q[len(q)-1-ii]
    xy=np.c_[q['x'],q['y']];inside=np.all((xy>=b[:2])&(xy<b[2:]),axis=1)
    ring=np.all((xy>=outer[:2])&(xy<outer[2:]),axis=1)&~inside
    f=q['frequency'][inside].astype(float)
    result=dict(qualified=False,reason='too_few_frequency_pixels',pixels=len(f),
                ring_pixels=int(ring.sum()),bbox=b.tolist(),frequency_hz=None,
                native_out_of_range_events_excluded=excluded)
    if not len(f):return result
    bins=np.floor(np.log(f/60.)/np.log(1.02)).astype(int)
    hist=np.bincount(bins,minlength=153);support=np.convolve(hist,np.ones(3,dtype=int),'same')
    k=int(np.argmax(support));keep=abs(bins-k)<=1;mode=float(np.median(f[keep]))
    relative=np.abs(np.log(q['frequency'].astype(float)/mode))<=np.log(1.02)*1.5
    selected=int(np.sum(inside&relative));background=int(np.sum(ring&relative))
    dominance=float(keep.mean());density=selected/area;bg_density=background/ring_area
    ok=len(f)>=16 and int(keep.sum())>=16 and dominance>=.35 and bg_density<density*.5
    result.update(qualified=bool(ok),reason='qualified_flicker' if ok else 'weak_or_common_mode',
        frequency_hz=mode,mode_pixels=int(keep.sum()),mode_fraction=dominance,
        matching_roi_pixels=selected,matching_ring_pixels=background,
        roi_density=density,ring_density=bg_density,harmonic_ambiguous=True)
    return result

class NativeFeatures:
    def __init__(self,w,h):
        self.w,self.h=w,h;self.buffer=deque();self.last_t=None
        self.algorithms={n:engine(w,h,n) for n in PARAMS['configurations']}
        self.outputs={n:a.get_empty_output_buffer() for n,a in self.algorithms.items()}
        self.frequency={n:deque() for n in self.algorithms}
    def feed(self,e,t):
        assert self.last_t is None or t>self.last_t
        assert not len(e) or (e['t'].min()>=round((t-.01)*1e6) and e['t'].max()<round(t*1e6))
        self.last_t=t;self.buffer.append((t,e.copy()))
        while self.buffer and self.buffer[0][0]<t-.160001:self.buffer.popleft()
        for name,a in self.algorithms.items():
            q=checked_process(a,e,self.outputs[name],self.w,self.h)
            self.frequency[name].append((t,q))
            while self.frequency[name] and self.frequency[name][0][0]<=t-.020001:self.frequency[name].popleft()
    def publish(self,bbox,t,velocity):
        assert t==self.last_t
        result={name:summarize(np.concatenate([q for _,q in data]),bbox,t,self.w,self.h)
                for name,data in self.frequency.items()}
        # Translation-compensated replay at the same cutoff. Reset the SDK
        # engine because per-window integer warp changes pixel coordinates;
        # carrying native states through such a warp would mix identities.
        e=np.concatenate([e for _,e in self.buffer]);e=e[e['t']>=round(t*1e6)-160000].copy()
        age=t-e['t']/1e6;xy=np.rint(np.c_[e['x'],e['y']]+age[:,None]*np.asarray(velocity)).astype(int)
        keep=np.all((xy>=0)&(xy<[self.w,self.h]),axis=1);e=e[keep];xy=xy[keep]
        e['x']=xy[:,0];e['y']=xy[:,1]
        a=engine(self.w,self.h,'strict7');o=a.get_empty_output_buffer()
        q=checked_process(a,e,o,self.w,self.h)
        result['aligned_strict7']=summarize(q,bbox,t,self.w,self.h)
        return result

def events(times,pixels,polarities):
    e=np.zeros(len(times),sdk_base.EventCD);e['t']=times;e['x']=pixels[:,0];e['y']=pixels[:,1];e['p']=polarities
    return np.sort(e,order='t')

def synthetic(kind,seed,w=128,h=96,duration=.4):
    rng=np.random.default_rng(79000+seed);freq=[83.,167.,287.,523.][seed%4]
    roi=np.array([35.,25.,93.,71.]);xx,yy=np.meshgrid(np.arange(43,86,3),np.arange(33,64,3));pixels=np.c_[xx.ravel(),yy.ravel()]
    if kind=='common_flicker':
        xx,yy=np.meshgrid(np.arange(10,119,3),np.arange(8,89,3));pixels=np.c_[xx.ravel(),yy.ravel()]
    ts=[];xy=[];pp=[]
    for pix in pixels:
        if kind in ['tone','common_flicker','moving_tone']:
            times=np.arange(.002,duration,1/(2*freq));pol=np.arange(len(times))%2
            times=times+rng.normal(0,8e-6,len(times))
        elif kind=='poisson':
            times=np.cumsum(rng.exponential(1/(2*freq),int(duration*4*freq+20)));times=times[times<duration];pol=rng.integers(0,2,len(times))
        elif kind=='colored':
            n=int(duration*2*freq);periods=(1/(2*freq))*np.exp(np.cumsum(rng.normal(0,.15,n)))
            times=np.cumsum(periods);times=times[times<duration];pol=np.arange(len(times))%2
        elif kind=='burst':
            times=.2+rng.uniform(-.001,.001,100);times.sort();pol=rng.integers(0,2,len(times))
        elif kind=='switch':
            times=np.r_[np.arange(.002,.2,1/(2*freq)),np.arange(.2,duration,1/(2*(freq*1.5)))];pol=np.arange(len(times))%2
        else:raise ValueError(kind)
        points=np.tile(pix,(len(times),1)).astype(float)
        if kind=='moving_tone':points[:,0]+=(times-duration)*100.
        keep=np.all((points>=0)&(points<[w,h]),axis=1)&(times>=0)&(times<duration)
        ts.extend(np.rint(times[keep]*1e6).astype(int));xy.extend(np.rint(points[keep]).astype(int));pp.extend(pol[keep])
    return events(np.asarray(ts),np.asarray(xy),np.asarray(pp)),roi,freq
