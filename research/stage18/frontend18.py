"""Incremental full-frame spectrum and bounded measurement-only publication."""
from pathlib import Path
import importlib.util,time
import numpy as np
from scipy.ndimage import label

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('stage17_frozen_frontend',ROOT/'research/stage17/frontend.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
PARAMS=dict(old.PARAMS,track_keep_s=.16,transfer_after_s=.08,scale_factor=2.,
    association_floor_px=50.,association_extent_fraction=.4)


class IncrementalSpectrum(old.SparseSpectrum):
    """400-bin uint32 ring; each disjoint20ms event batch is counted once."""
    def __init__(self,w,h):
        super().__init__(w,h)
        self.ring=np.zeros((self.rows*self.cols,self.n),np.uint32)
        self.totals=np.zeros(self.rows*self.cols,np.int64)
        self.end_bin=None;self.ingested_events=0

    def update(self,events,t):
        end_us=round(t*1e6)
        if end_us%self.bin_us:raise ValueError('Cutoff must align with250us bins')
        end=end_us//self.bin_us;width=PARAMS['step_us']//self.bin_us
        if self.end_bin is not None and end<=self.end_bin:raise ValueError('Cutoffs must strictly increase')
        if self.end_bin is not None and end-self.end_bin<width:raise ValueError('20ms input slices must not overlap')
        if self.end_bin is None or end-self.end_bin>=self.n:
            self.ring.fill(0);self.totals.fill(0)
        else:
            expired=np.arange(self.end_bin,end)%self.n
            self.totals-=self.ring[:,expired].sum(axis=1,dtype=np.int64)
            self.ring[:,expired]=0
        # Input contract is the newest disjoint20ms slice. Ignore injected future/old events.
        e=events
        if len(e) and (int(e['t'].min())<end_us-PARAMS['step_us'] or int(e['t'].max())>=end_us):
            e=e[(e['t']>=end_us-PARAMS['step_us'])&(e['t']<end_us)]
        if len(e):
            block=e['y'].astype(np.int32)//self.b*self.cols+e['x']//self.b
            offset=((e['t']-(end_us-PARAMS['step_us']))//self.bin_us).astype(np.int32)
            hist=np.bincount(block*width+offset,minlength=len(self.totals)*width).reshape(-1,width)
            columns=np.arange(end-width,end)%self.n
            self.ring[:,columns]=hist.astype(np.uint32)
            self.totals+=hist.sum(axis=1,dtype=np.int64)
        self.end_bin=end;self.ingested_events+=len(e)

    def step(self,unused_history,t,roi=None):
        if roi is not None:raise ValueError('This equivalence branch uses full-frame evidence')
        if self.end_bin!=round(t*1e6)//self.bin_us:raise ValueError('Call update before spectrum evaluation')
        active=np.flatnonzero(self.totals>=PARAMS['min_events'])
        quality=np.zeros(self.rows*self.cols,np.float32);hz=quality.copy()
        if len(active):
            order=(np.arange(self.n)+self.end_bin)%self.n
            series=self.ring[active[:,None],order[None,:]].astype(np.float32)
            series-=series.mean(1,keepdims=True)
            power=np.abs(np.fft.rfft(series*self.hann,axis=1))**2;power=power[:,self.band]
            peak=np.argmax(power,axis=1)
            quality[active]=power[np.arange(len(active)),peak]/np.maximum(np.median(power,axis=1),1e-9)
            hz[active]=self.freq[self.band][peak]
        quality=quality.reshape(self.rows,self.cols);hz=hz.reshape(self.rows,self.cols)
        mask=quality>=PARAMS['quality'];components,n=label(mask,structure=np.ones((3,3)))
        regions=[]
        for k in range(1,n+1):
            yy,xx=np.nonzero(components==k)
            if len(xx)>=2:regions.append(dict(blocks=len(xx),center=[float((xx.mean()+.5)*self.b),float((yy.mean()+.5)*self.b)]))
        return dict(quality=quality,hz=hz,mask=mask,active_blocks=len(active),
            eligible_blocks=int(mask.sum()),regions=regions,events=int(self.totals.sum()))


class ProposalGeometry(old.GeometryFrontend):
    def detect_prepared(self,events,t,prepared):
        observation,total=super().detect_prepared(events,t,prepared)
        self.last_proposal=observation
        return observation,total


class IncrementalFrontend(old.Frontend):
    def __init__(self,w,h):
        super().__init__(w,h,'global')
        self.spectral=IncrementalSpectrum(w,h);self.geometry=ProposalGeometry(w,h)

    def advance(self,current,t,ready=True):
        started=time.perf_counter();self.spectral.update(current,t);ingest=(time.perf_counter()-started)*1000
        if not ready:
            return dict(observation=dict(t=t,detected=False,rotors=[]),proposal=None,
                timing_ms=dict(histogram_ingest=ingest,total=ingest),search='warming')
        q=super().step(None,current,t)
        q['proposal']=self.geometry.last_proposal
        q['timing_ms']['histogram_ingest']=ingest;q['timing_ms']['total']+=ingest
        return q


class TrackPublisher:
    """Latch confirmed tracks160ms; no position is emitted without current geometry."""
    def __init__(self):
        self.last=None;self.velocity=np.zeros(2);self.pending=None

    def _measurement(self,o,t):
        box=np.array(o['bbox'],float)
        return dict(t=t,center=np.array(o['center'],float),extent=float(max(box[2:]-box[:2])))

    def _close(self,a,b,predict=False):
        center=a['center']+(self.velocity*(b['t']-a['t']) if predict else 0)
        scale=max(a['extent'],b['extent'])/max(1.,min(a['extent'],b['extent']))
        distance=np.linalg.norm(center-b['center'])
        return scale<=PARAMS['scale_factor'] and distance<max(PARAMS['association_floor_px'],max(a['extent'],b['extent'])*PARAMS['association_extent_fraction'])

    def apply(self,proposal,t):
        if proposal is None or not proposal['detected']:
            self.pending=None
            return dict(observation=dict(t=t,detected=False,rotors=[],confidence=0.),status='no_current_measurement')
        b=self._measurement(proposal,t);gap=float('inf') if self.last is None else t-self.last['t']
        locked=self.last is not None and 0<gap<=PARAMS['track_keep_s'] and self._close(self.last,b,predict=True)
        if locked:
            status='measured_locked';self.pending=None;published=True
        else:
            prior=self.pending
            consecutive=prior is not None and 0<t-prior['t']<=.025 and self._close(prior,b)
            b['count']=prior['count']+1 if consecutive else 1
            self.pending=b
            published=b['count']>=PARAMS['publish_confirm_frames'] and gap>=PARAMS['transfer_after_s']
            status='confirmed_hypothesis' if published else 'initializing_hypothesis'
        if published:
            if self.last is not None and 0<gap<=PARAMS['track_keep_s']:
                self.velocity=.7*self.velocity+.3*np.clip((b['center']-self.last['center'])/gap,-2500,2500)
            else:self.velocity[:]=0
            self.last=b
            return dict(observation=proposal,status=status)
        return dict(observation=dict(t=t,detected=False,rotors=[],confidence=0.),status=status)
