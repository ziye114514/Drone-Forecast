"""Sparse frequency evidence + reused geometry/state protection, RAW only."""
from pathlib import Path
import sys,time
import cv2,numpy as np
from scipy.ndimage import label

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage9'),str(ROOT/'research/stage8')]
from recovery import RecoveryFrontend
from polarity import counts
from event_drone_forecast.rotor_sensing import PaperPerception

PARAMS=dict(step_us=20000,window_us=100000,bin_us=250,block=8,
    min_hz=60.,max_hz=1600.,min_events=60,quality=20.,mask_dilate_blocks=3,
    periodic_coverage=.15,frequency_mad_fraction=.35,global_period_us=100000,
    roi_expiry_s=.16,roi_padding_fraction=.5,roi_min_padding_px=40,
    publish_confirm_frames=3)


def inside(events,roi):
    if roi is None:return events
    x0,y0,x1,y1=roi
    return events[(events['x']>=x0)&(events['x']<x1)&(events['y']>=y0)&(events['y']<y1)]


class SparseSpectrum:
    def __init__(self,w,h,bin_us=250):
        self.w,self.h,self.b=w,h,PARAMS['block'];self.rows=(h+self.b-1)//self.b;self.cols=(w+self.b-1)//self.b
        self.bin_us=bin_us;self.n=PARAMS['window_us']//bin_us
        nyquist=1e6/(2*bin_us)
        if PARAMS['max_hz']>=nyquist:raise ValueError('Configured upper band must be strictly below Nyquist')
        self.freq=np.fft.rfftfreq(self.n,bin_us/1e6)
        self.band=(self.freq>=PARAMS['min_hz'])&(self.freq<=PARAMS['max_hz'])
        self.hann=np.hanning(self.n).astype(np.float32)

    def step(self,events,t,roi=None):
        start=round(t*1e6)-PARAMS['window_us'];end=round(t*1e6)
        e=inside(events,roi);e=e[(e['t']>=start)&(e['t']<end)]
        block=e['y'].astype(np.int32)//self.b*self.cols+e['x']//self.b
        total=np.bincount(block,minlength=self.rows*self.cols)
        active=np.flatnonzero(total>=PARAMS['min_events'])
        quality=np.zeros(self.rows*self.cols,np.float32);hz=quality.copy()
        if len(active):
            mapping=np.full(self.rows*self.cols,-1,np.int32);mapping[active]=np.arange(len(active))
            index=mapping[block];ok=index>=0
            ts=((e['t'][ok]-start)//self.bin_us).astype(np.int32)
            series=np.bincount(index[ok]*self.n+ts,minlength=len(active)*self.n).reshape(len(active),self.n).astype(np.float32)
            series-=series.mean(1,keepdims=True)
            power=np.abs(np.fft.rfft(series*self.hann,axis=1))**2
            power=power[:,self.band];peak=np.argmax(power,axis=1)
            quality[active]=power[np.arange(len(active)),peak]/np.maximum(np.median(power,axis=1),1e-9)
            hz[active]=self.freq[self.band][peak]
        quality=quality.reshape(self.rows,self.cols);hz=hz.reshape(self.rows,self.cols)
        mask=quality>=PARAMS['quality'];components,n=label(mask,structure=np.ones((3,3)))
        regions=[]
        for k in range(1,n+1):
            yy,xx=np.nonzero(components==k)
            if len(xx)>=2:regions.append(dict(blocks=len(xx),center=[float((xx.mean()+.5)*self.b),float((yy.mean()+.5)*self.b)]))
        return dict(quality=quality,hz=hz,mask=mask,active_blocks=len(active),
                    eligible_blocks=int(mask.sum()),regions=regions,events=len(e))


class GeometryFrontend(RecoveryFrontend):
    def __init__(self,w,h):
        super().__init__(w,h,'strict10');self.allowed=np.zeros((h,w),np.float32)
        self.spectrum=None;self.raw_candidates=[]

    def feature(self,on,off):
        # Per-polarity coincidence is deliberately not required in this branch.
        return (on+off)*self.allowed

    def _candidates(self,*args,**kwargs):
        found=PaperPerception._candidates(self,*args,**kwargs);kept=[]
        spec=self.spectrum;b=PARAMS['block'];h,w=spec['mask'].shape
        for r in found:
            x0,y0,x1,y1=r['bbox'];x0=max(0,int(x0//b));y0=max(0,int(y0//b))
            x1=min(w,int(np.ceil(x1/b)));y1=min(h,int(np.ceil(y1/b)))
            mask=spec['mask'][y0:y1,x0:x1];freq=spec['hz'][y0:y1,x0:x1][mask]
            coverage=float(mask.mean()) if mask.size else 0.
            med=float(np.median(freq)) if len(freq) else None
            dispersion=float(np.median(abs(freq-med))/max(med,1.)) if len(freq) else None
            accept=bool(coverage>=PARAMS['periodic_coverage'] and dispersion is not None and dispersion<=PARAMS['frequency_mad_fraction'])
            row=dict(cx=float(r['center'][0]),cy=float(r['center'][1]),major=float(r['major']),minor=float(r['minor']),
                angle=float(r['angle']),mode=r['mode'],periodic_coverage=coverage,block_frequency_hz=med,
                block_frequency_mad_fraction=dispersion,accepted=accept)
            self.raw_candidates.append(row)
            if accept:
                self.candidates.append(row);kept.append(r)
        return kept


class Frontend:
    def __init__(self,w,h,policy='roi'):
        if policy not in ['global','roi']:raise ValueError(policy)
        self.w,self.h,self.policy=w,h,policy;self.spectral=SparseSpectrum(w,h)
        self.geometry=GeometryFrontend(w,h);self.last_global=None
        self.center=None;self.velocity=np.zeros(2);self.last_seen=None;self.box=None
        self.confirm=0;self.pending_center=None;self.pending_t=None

    def _roi(self,t):
        if self.policy=='global' or self.last_seen is None or t-self.last_seen>PARAMS['roi_expiry_s']:return None
        if self.last_global is None or round(t*1e6)-self.last_global>=PARAMS['global_period_us']:return None
        box=self.box.copy();dt=t-self.last_seen;box+=np.r_[self.velocity,self.velocity]*dt
        size=box[2:]-box[:2];pad=np.maximum(size*PARAMS['roi_padding_fraction'],PARAMS['roi_min_padding_px'])
        lo=np.maximum(np.floor(box[:2]-pad),0).astype(int)
        hi=np.minimum(np.ceil(box[2:]+pad),[self.w,self.h]).astype(int)
        return [int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]

    def step(self,history,current,t):
        began=time.perf_counter();roi=self._roi(t)
        if roi is None:self.last_global=round(t*1e6)
        spec=self.spectral.step(history,t,roi);after_spectrum=time.perf_counter()
        mask=cv2.dilate(np.uint8(spec['mask']),(np.ones((PARAMS['mask_dilate_blocks'],)*2,np.uint8)))
        allowed=np.repeat(np.repeat(mask,PARAMS['block'],0),PARAMS['block'],1)[:self.h,:self.w].astype(np.float32)
        if roi is not None:
            x0,y0,x1,y1=roi;crop=np.zeros_like(allowed);crop[y0:y1,x0:x1]=allowed[y0:y1,x0:x1];allowed=crop
        e=inside(current,roi);prepared=counts(e,self.w,self.h)
        prepared['on']/=PARAMS['step_us']/10000;prepared['off']/=PARAMS['step_us']/10000
        g=self.geometry;g.allowed=allowed;g.spectrum=spec;g.raw_candidates=[]
        proposal,_=g.detect_prepared(e,t,prepared)
        if proposal['detected']:
            p=np.array(proposal['center']);extent=max(np.array(proposal['bbox'][2:])-proposal['bbox'][:2])
            consecutive=self.pending_t is not None and 0<t-self.pending_t<=.025
            close=self.pending_center is not None and np.linalg.norm(p-self.pending_center)<max(50.,extent*.4)
            self.confirm=self.confirm+1 if consecutive and close else 1
            self.pending_center=p;self.pending_t=t
        else:self.confirm=0;self.pending_center=None;self.pending_t=None
        published=proposal['detected'] and self.confirm>=PARAMS['publish_confirm_frames']
        observation=proposal if published else dict(t=t,detected=False,rotors=[],confidence=0.,n_events=len(current))
        if published:
            p=np.array(proposal['center'])
            if self.last_seen is not None and 0<t-self.last_seen<=PARAMS['roi_expiry_s']:
                self.velocity=.7*self.velocity+.3*np.clip((p-self.center)/(t-self.last_seen),-2500,2500)
            else:self.velocity[:]=0
            self.center=p;self.box=np.array(proposal['bbox']);self.last_seen=t
        return dict(observation=observation,proposal_detected=proposal['detected'],confirmation=self.confirm,
            search='global' if roi is None else 'roi',roi=roi,active_blocks=spec['active_blocks'],
            eligible_blocks=spec['eligible_blocks'],periodic_regions=spec['regions'],periodic_present=bool(spec['regions']),
            candidates=g.candidates,raw_candidates=g.raw_candidates,audit=g.audit,
            timing_ms=dict(spectrum=(after_spectrum-began)*1000,total=(time.perf_counter()-began)*1000))
