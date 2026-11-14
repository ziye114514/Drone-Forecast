"""Causal, center-free ON/OFF activity and periodic force *proxies*.

Neither event rate nor an unlabelled spectral peak is mechanical RPM. These
features deliberately remain separate so prediction ablations can test that
assumption instead of silently treating it as a calibrated thrust sensor.
"""
from collections import deque
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage66')]
from periodicity66 import spectrum,peak,PARAMS as OLD

PARAMS=dict(window_us=160000,bin_us=100,grid=[6,3],max_events=450000,
    body_confidence_min=.45,min_events=100,tone_fraction_min=.15,
    phase_coherence_min=.8,background_fraction_max=.5,reference_samples=5,
    minimum_time_span_us=96000,
    ratio_bounds=[.2,5.],proxy_expiry_s=.16,
    density='20ms balanced ON/OFF count / current ROI area, minus expanded-ring '
            'balanced count / ring area. Per-event past-velocity translation. '
            'Normalise by first five qualified past/current samples, then square.',
    frequency='160ms 6x3 polarity spectra; top five SNR>=8 peaks, phase coherence '
              'and sinusoidal explained energy. Reject spatially common ring tone. '
              'Weighted median of tile frequency squared, not a four-motor sum.',
    interpretation='Effective image-domain input only. Contrast, pose, apparent '
                   'area, camera motion and harmonics can confound it.')


def tone(signals,f):
    a=np.asarray(signals,float);a=a-a.mean(axis=-1,keepdims=True)
    carrier=np.exp(-2j*np.pi*f*(np.arange(a.shape[-1])+.5)/10000.)
    middle=a.shape[-1]//2
    early=(a[...,:middle]*carrier[:middle]).sum(axis=-1)
    late=(a[...,middle:]*carrier[middle:]).sum(axis=-1)
    energy=float(np.sum(a*a))
    fraction=float(2*np.sum(abs(early+late)**2)/max(a.shape[-1]*energy,1e-9))
    coherence=float(np.sum(abs(early+late))/max(np.sum(abs(early)+abs(late)),1e-9))
    return dict(frequency_hz=float(f),tone_fraction=fraction,phase_coherence=coherence)


class Activity:
    def __init__(self,width=1280,height=720):self.buffer=deque();self.width=width;self.height=height;self.last_t=None
    def step(self,events,bbox,t,velocity):
        if self.last_t is not None:assert t>self.last_t
        assert not len(events) or (events['t'].min()>=round((t-.02)*1e6) and events['t'].max()<round(t*1e6))
        self.last_t=t
        self.buffer.append((t,events.copy()))
        while self.buffer and self.buffer[0][0]<t-.160001:self.buffer.popleft()
        if bbox is None:return dict(valid=False,reason='no_current_NN_identity')
        b=np.asarray(bbox,float).copy();b[:2]=np.maximum(b[:2],0);b[2:]=np.minimum(b[2:],[self.width,self.height]);size=b[2:]-b[:2]
        if min(size)<12:return dict(valid=False,reason='small_ROI')
        all_e=np.concatenate([q[1] for q in self.buffer]);start=round(t*1e6)-160000
        all_e=all_e[(all_e['t']>=start)&(all_e['t']<round(t*1e6))]
        count=len(all_e);stride=max(1,int(np.ceil(count/PARAMS['max_events'])));e=all_e[::stride]
        age=t-e['t']/1e6;v=np.asarray(velocity,float)
        xy=np.c_[e['x'],e['y']].astype(float)+age[:,None]*v
        inside=np.all((xy>=b[:2])&(xy<b[2:]),axis=1)
        pad=np.maximum(size*.25,12);outer=np.r_[b[:2]-pad,b[2:]+pad]
        outer[:2]=np.maximum(outer[:2],0);outer[2:]=np.minimum(outer[2:],[self.width,self.height])
        ring=np.all((xy>=outer[:2])&(xy<outer[2:]),axis=1)&~inside
        area=float(np.prod(size));ring_area=float(np.prod(outer[2:]-outer[:2])-area)
        current=e['t']>=round(t*1e6)-20000
        def balanced(mask):
            on=int(np.sum(mask&(e['p']>0)));off=int(np.sum(mask&(e['p']<=0)))
            return 2*min(on,off)*stride,on*stride,off*stride
        balanced_roi,on,off=balanced(inside&current);balanced_bg,bg_on,bg_off=balanced(ring&current)
        density=max(0.,balanced_roi/area-balanced_bg/ring_area)/.02
        n=1600;gx,gy=PARAMS['grid'];bins=((e['t']-start)//100).astype(int)
        ix=np.minimum(gx-1,((xy[inside,0]-b[0])/size[0]*gx).astype(int))
        iy=np.minimum(gy-1,((xy[inside,1]-b[1])/size[1]*gy).astype(int))
        cell=iy*gx+ix;pol=(e['p']>0).astype(int)
        signals=np.bincount((cell*2+pol[inside])*n+bins[inside],minlength=gx*gy*2*n).reshape(gx*gy,2,n)
        background=np.bincount(pol[ring]*n+bins[ring],minlength=2*n).reshape(2,n)
        f,p=spectrum(signals);candidates=[];accepted=[]
        for j,a in enumerate(signals):
            if a.sum()<PARAMS['min_events']:continue
            q=peak(f,p[j])
            if q is None:continue
            noise=max(float(np.median(p[j,(f>=60)&(f<=1200)])),1e-12)
            best=None
            for freq in q['top_peaks_hz']:
                k=int(np.argmin(abs(f-freq)));snr=float(p[j,k]/noise)
                if snr<8:continue
                tt=tone(a,freq);bg=tone(background,freq)
                tt.update(tile=j,snr=snr,background_tone_fraction=bg['tone_fraction'])
                tt['qualified']=bool(tt['tone_fraction']>=PARAMS['tone_fraction_min'] and
                    tt['phase_coherence']>=PARAMS['phase_coherence_min'] and
                    bg['tone_fraction']<tt['tone_fraction']*PARAMS['background_fraction_max'] and
                    len(e)>0 and e['t'].max()-e['t'].min()>=PARAMS['minimum_time_span_us'])
                candidates.append(tt)
                if tt['qualified'] and (best is None or tt['tone_fraction']>best['tone_fraction']):best=tt
            if best is not None:accepted.append(best)
        value=None
        if accepted:
            values=np.array([q['frequency_hz']**2 for q in accepted]);weights=np.array([q['tone_fraction'] for q in accepted])
            order=np.argsort(values);weights=weights[order];values=values[order]
            value=float(values[np.searchsorted(np.cumsum(weights),weights.sum()/2)])
        return dict(valid=True,reason='current_NN_ROI_activity',bbox=b.tolist(),
            density=density,roi_on=on,roi_off=off,ring_on=bg_on,ring_off=bg_off,
            roi_area=area,ring_area=ring_area,frequency_squared=value,
            periodic_tiles=len(accepted),spectral_candidates=candidates,
            raw_events=count,retained_events=len(e),stride=stride,
            velocity=v.tolist(),mechanical_RPM_or_metric_thrust=False)


class Reference:
    def __init__(self):self.samples=[];self.value=None;self.last=None;self.ratio=None
    def step(self,t,value):
        current=value is not None and np.isfinite(value) and value>0
        if current:
            if self.value is None:
                self.samples.append(float(value))
                if len(self.samples)>=PARAMS['reference_samples']:self.value=float(np.median(self.samples))
            if self.value is not None:
                ratio=float(value/self.value)
                if PARAMS['ratio_bounds'][0]<=ratio<=PARAMS['ratio_bounds'][1]:self.last=t;self.ratio=ratio
                else:current=False
        active=self.last is not None and t-self.last<=PARAMS['proxy_expiry_s']+1e-8
        return dict(active=active,current=bool(current and self.value is not None and self.last==t),
                    ratio=self.ratio if active else None,reference=self.value,
                    age_s=None if self.last is None else t-self.last)
