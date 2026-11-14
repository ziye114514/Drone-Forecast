"""Paper-style ON-after-OFF histogram plus explicit unique-pixel/ring controls."""
from collections import deque
import numpy as np

PARAMS=dict(window_us=100000,bin_us=100,bins=256,hot_pixel_percentile=70.,
    hot_pixel_population='Nonzero per-pixel counts in current20ms ROI; percentile choice explicit.',
    minimum_pixels=16,minimum_peak_fraction=.35,maximum_background_density_ratio=.5,
    min_hz=60.,max_hz=1200.,interval='ON after OFF; duty cycle and harmonic ambiguity, not mechanical RPM.')


class PeriodHistogram:
    def __init__(self,w,h):
        self.w,self.h=w,h;self.last_off=np.full(w*h,-1,np.int64);self.polarity=np.full(w*h,-1,np.int8);self.periods=deque()

    def step(self,e,bbox,t):
        end=round(t*1e6)
        assert not len(e) or e['t'].max()<end
        if bbox is None:
            self.periods.clear();self.last_off.fill(-1);self.polarity.fill(-1)
            return dict(raw_qualified=False,guarded_qualified=False,reason='no_ROI',frequency_hz=None,pixels=0)
        b=np.array(bbox,float);b[:2]=np.maximum(b[:2],0);b[2:]=np.minimum(b[2:],[self.w,self.h]);size=b[2:]-b[:2]
        if min(size)<12:return dict(raw_qualified=False,guarded_qualified=False,reason='small_ROI',frequency_hz=None,pixels=0)
        pad=np.maximum(size*.25,12);outer=np.r_[np.maximum(b[:2]-pad,0),np.minimum(b[2:]+pad,[self.w,self.h])]
        inside=(e['x']>=b[0])&(e['x']<b[2])&(e['y']>=b[1])&(e['y']<b[3])
        relevant=(e['x']>=outer[0])&(e['x']<outer[2])&(e['y']>=outer[1])&(e['y']<outer[3])
        pixel=e['y'].astype(np.int64)*self.w+e['x'];unique,counts=np.unique(pixel[inside],return_counts=True)
        if len(counts):hot=unique[counts>=np.percentile(counts,70.)]
        else:hot=np.empty(0,int)
        allowed=set(hot.tolist());new=[]
        for ev,pix,in_roi in zip(e[relevant],pixel[relevant],inside[relevant]):
            stamp=int(ev['t']);pol=int(ev['p']>0)
            if pol and self.polarity[pix]==0 and self.last_off[pix]>=0:
                delta=stamp-self.last_off[pix]
                if 0<delta<25600 and (not in_roi or pix in allowed):new.append((stamp,int(pix),int(delta)))
            if not pol:self.last_off[pix]=stamp
            self.polarity[pix]=pol
        self.periods.extend(new)
        while self.periods and self.periods[0][0]<end-100000:self.periods.popleft()
        if not self.periods:return dict(raw_qualified=False,guarded_qualified=False,reason='no_intervals',frequency_hz=None,pixels=0)
        values=np.asarray(self.periods,dtype=np.int64);pix=values[:,1];xy=np.c_[pix%self.w,pix//self.w]
        roi=np.all((xy>=b[:2])&(xy<b[2:]),axis=1);ring=np.all((xy>=outer[:2])&(xy<outer[2:]),axis=1)&~roi
        bins=values[:,2]//100;hist=np.bincount(bins[roi],minlength=256);k=int(np.argmax(hist));f=1000000./((k+.5)*100)
        raw=bool(hist[k]>0 and 60<=f<=1200)
        # Pooling events would let one hot pixel dominate. Use the latest
        # interval of each pixel for admission, while retaining raw paper argmax.
        _,ii=np.unique(pix[::-1],return_index=True);idx=len(pix)-1-ii
        chosen=idx[roi[idx]];near=abs(bins-k)<=1;support=int(np.sum(near[chosen]));pixels=len(chosen)
        bg=int(np.sum(ring[idx]&near[idx]));area=float(np.prod(size));ring_area=float(np.prod(outer[2:]-outer[:2])-area)
        fraction=support/max(1,pixels);guard=bool(raw and support>=16 and fraction>=.35 and bg/ring_area<.5*support/area)
        return dict(raw_qualified=raw,guarded_qualified=guard,reason='qualified' if guard else 'insufficient_or_common_support',
            frequency_hz=f if raw else None,paper_argmax_hz=f,pixels=pixels,peak_pixels=support,peak_fraction=fraction,
            ring_peak_pixels=bg,interval_count=int(roi.sum()),raw_peak_bin=k,harmonic_or_duty_cycle_ambiguous=True)
