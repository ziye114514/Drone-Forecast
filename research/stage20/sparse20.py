"""Sparse disjoint chunk histograms; exact100ms spectrum without dense23MB ring."""
from pathlib import Path
from collections import deque
import importlib.util
import numpy as np
from scipy.ndimage import label

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('reused_stage18',ROOT/'research/stage18/frontend18.py')
inc=importlib.util.module_from_spec(spec);spec.loader.exec_module(inc)
PARAMS=inc.PARAMS


class SparseWindow(inc.old.SparseSpectrum):
    def __init__(self,w,h):
        super().__init__(w,h);self.chunks=deque();self.totals=np.zeros(self.rows*self.cols,np.int64)
        self.end_us=None;self.ingested_events=0

    def update(self,events,t):
        end=round(t*1e6);width=PARAMS['step_us']//self.bin_us
        if end%self.bin_us:raise ValueError('Cutoff must align with250us bins')
        if self.end_us is not None and end-self.end_us<PARAMS['step_us']:raise ValueError('20ms input slices must not overlap')
        start=end-PARAMS['step_us'];window_start=end-PARAMS['window_us']
        while self.chunks and self.chunks[0]['end']<=window_start:
            chunk=self.chunks.popleft();self.totals[chunk['active']]-=chunk['counts'].sum(axis=1,dtype=np.int64)
        if self.chunks and self.chunks[0]['start']<window_start:
            chunk=self.chunks[0];cut=(window_start-chunk['start'])//self.bin_us
            self.totals[chunk['active']]-=chunk['counts'][:,:cut].sum(axis=1,dtype=np.int64)
            chunk['counts']=chunk['counts'][:,cut:];chunk['start']=window_start
        e=events
        if len(e) and (int(e['t'].min())<start or int(e['t'].max())>=end):e=e[(e['t']>=start)&(e['t']<end)]
        active=np.empty(0,np.int32);hist=np.empty((0,width),np.uint32)
        if len(e):
            block=e['y'].astype(np.int32)//self.b*self.cols+e['x']//self.b
            totals=np.bincount(block,minlength=len(self.totals));active=np.flatnonzero(totals).astype(np.int32)
            mapping=np.full(len(self.totals),-1,np.int32);mapping[active]=np.arange(len(active))
            offsets=((e['t']-start)//self.bin_us).astype(np.int32)
            hist=np.bincount(mapping[block]*width+offsets,minlength=len(active)*width).reshape(len(active),width).astype(np.uint32)
            self.totals[active]+=totals[active]
        self.chunks.append(dict(start=start,end=end,active=active,counts=hist));self.end_us=end;self.ingested_events+=len(e)

    def step(self,unused,t,roi=None):
        if roi is not None:raise ValueError('Exact full-frame statistics only')
        if self.end_us!=round(t*1e6):raise ValueError('Update before spectrum')
        active=np.flatnonzero(self.totals>=PARAMS['min_events'])
        quality=np.zeros(len(self.totals),np.float32);hz=quality.copy()
        if len(active):
            mapping=np.full(len(self.totals),-1,np.int32);mapping[active]=np.arange(len(active))
            series=np.zeros((len(active),self.n),np.float32);window_start=self.end_us-PARAMS['window_us']
            for chunk in self.chunks:
                rows=mapping[chunk['active']];keep=rows>=0;offset=(chunk['start']-window_start)//self.bin_us
                series[rows[keep],offset:offset+chunk['counts'].shape[1]]=chunk['counts'][keep]
            series-=series.mean(1,keepdims=True)
            power=np.abs(np.fft.rfft(series*self.hann,axis=1))**2;power=power[:,self.band];peak=np.argmax(power,axis=1)
            quality[active]=power[np.arange(len(active)),peak]/np.maximum(np.median(power,axis=1),1e-9)
            hz[active]=self.freq[self.band][peak]
        quality=quality.reshape(self.rows,self.cols);hz=hz.reshape(self.rows,self.cols)
        mask=quality>=PARAMS['quality'];components,n=label(mask,structure=np.ones((3,3)));regions=[]
        for k in range(1,n+1):
            yy,xx=np.nonzero(components==k)
            if len(xx)>=2:regions.append(dict(blocks=len(xx),center=[float((xx.mean()+.5)*self.b),float((yy.mean()+.5)*self.b)]))
        return dict(quality=quality,hz=hz,mask=mask,active_blocks=len(active),eligible_blocks=int(mask.sum()),regions=regions,events=int(self.totals.sum()))

    def allocated_histogram_bytes(self):return sum(c['counts'].nbytes+c['active'].nbytes for c in self.chunks)+self.totals.nbytes


class SparseFrontend(inc.IncrementalFrontend):
    def __init__(self,w,h):
        inc.old.Frontend.__init__(self,w,h,'global')
        self.spectral=SparseWindow(w,h);self.geometry=inc.ProposalGeometry(w,h)
