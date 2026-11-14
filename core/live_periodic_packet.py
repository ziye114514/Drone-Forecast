"""Lossless transport reduction for the current stateless141 observer only.

Both79.summarize and141.measure_current filter to current20ms, discard invalid
frequencies, then use one latest admissible frequency per pixel. Keep all
current invalid-frequency records as well, preserving their exclusion count.
This packet must never replace input to a stateful frequency/phase engine.
"""
import numpy as np

def compact_current(q,t,width):
    if q is None:return None
    if not len(q):return q
    now=round(t*1e6);assert int(q['t'].max())<now
    current=q['t']>=now-20000
    allowed=(q['frequency']>=60.)&(q['frequency']<=1200.)
    good=np.flatnonzero(current&allowed)
    if len(good):
        pixels=q['y'][good].astype(np.int64)*width+q['x'][good]
        _,reverse_indices=np.unique(pixels[::-1],return_index=True)
        latest=good[len(good)-1-reverse_indices]
    else:latest=np.empty(0,np.int64)
    excluded=np.flatnonzero(current&~allowed)
    indices=np.sort(np.r_[latest,excluded])
    return q[indices]
