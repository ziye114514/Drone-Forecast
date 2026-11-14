"""Exact CPU joint native histogram and sparse NDFT inputs.

The native33,333us window is split at13,333us and16,667us, preserving both
halves and the last20ms. Integer counts/indices are exactly float32-representable
under the runtime's4M-event cap. No timestamps are binned for the spectrum.
"""
import cv2
import numpy as np


def prepare_histogram(events,begin,end,width,height,hist=None):
    if end-begin!=33333:raise ValueError('Hybrid histogram requires the native33333us window')
    size=width*height
    if 6*size>=2**24 or len(events)>=2**24:raise ValueError('Exact OpenCV histogram float32 bound exceeded')
    valid=(events['t']>=begin)&(events['t']<end)&(events['x']<width)&(events['y']<height)
    e=events if valid.all() else events[valid]
    t=e['t']-begin;p=e['y'].astype(np.int32)*width+e['x'].astype(np.int32)
    bins=(t>=13333).astype(np.int32)+(t>=16667).astype(np.int32)
    keys=(p+size*(2*bins+(e['p']>0))).astype(np.float32)
    if hist is None:hist=np.empty((6*size,1),np.float32)
    hist=cv2.calcHist([keys],[0],None,[6*size],[0.,float(6*size)],hist=hist,accumulate=False)
    h=hist.reshape(3,2,height,width)
    off=h[:,0].sum(0);on=h[:,1].sum(0)
    counts=np.stack([off,on]).astype(np.int32);total=counts[0]+counts[1]
    half=h[:2].sum(axis=(0,1)).astype(np.int64)
    local=cv2.boxFilter(total,cv2.CV_32S,(31,31),normalize=False,borderType=cv2.BORDER_REPLICATE)
    dense=(total>=6)&(total.astype(np.int64)*961>=2*local)
    eligible=dense&(half>=6)&(total-half>=6)
    pixels=np.flatnonzero(eligible).astype(np.int64)
    lengths=total.reshape(-1)[pixels].astype(np.int64);nt=half.reshape(-1)[pixels]
    selected=eligible.reshape(-1)[p]
    sp=p[selected];st=t[selected];pol=e['p'][selected]>0
    order=np.argsort(sp.astype(np.int64)*2+(st*2>=33333),kind='stable')
    times=st[order].astype(np.int64);polarity=pol[order].astype(np.int16)
    offsets=np.cumsum(lengths)-lengths
    ap=np.flatnonzero(total);activity=np.empty((len(ap),7),np.float32)
    activity[:,0]=ap;activity[:,1]=on.reshape(-1)[ap];activity[:,2]=off.reshape(-1)[ap]
    activity[:,3]=local.reshape(-1)[ap].astype(np.float32)/np.float32(961.)
    activity[:,4]=dense.reshape(-1)[ap]
    activity[:,5]=h[1:,1].sum(0).reshape(-1)[ap];activity[:,6]=h[1:,0].sum(0).reshape(-1)[ap]
    return dict(counts=counts,pixels=pixels,lengths=lengths,ntrain=nt,offsets=offsets,
        times=times,polarity=polarity,activity=activity,hist=hist,events=len(e))
