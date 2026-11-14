import numpy as np
from sparse20 import SparseWindow,inc


def run():
    rng=np.random.default_rng(2020);dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')])
    e=np.zeros(120000,dtype);e['x']=rng.integers(0,64,len(e));e['y']=rng.integers(0,48,len(e));e['p']=1;e['t']=np.sort(rng.integers(0,900000,len(e)))
    sparse=SparseWindow(64,48);dense=inc.IncrementalSpectrum(64,48);reference=inc.old.SparseSpectrum(64,48)
    seen=[];n=0
    # Include skipped30ms gap: retained oldest chunk requires partial trimming.
    cutoffs=list(range(20000,180001,20000))+list(range(210000,890001,20000))
    for hi in cutoffs:
        current=e[(e['t']>=hi-20000)&(e['t']<hi)];seen.append(current)
        sparse.update(current,hi/1e6);dense.update(current,hi/1e6)
        history=np.concatenate(seen);history=history[(history['t']>=hi-100000)&(history['t']<hi)]
        a=sparse.step(None,hi/1e6);b=dense.step(None,hi/1e6);c=reference.step(history,hi/1e6)
        for key in ['quality','hz','mask']:
            np.testing.assert_array_equal(a[key],b[key]);np.testing.assert_array_equal(a[key],c[key])
        for key in ['regions','events','active_blocks','eligible_blocks']:assert a[key]==b[key]==c[key]
        n+=1
    empty=np.empty(0,dtype);sparse.update(empty,1.1);assert sparse.step(None,1.1)['events']==0
    a=SparseWindow(64,48);b=SparseWindow(64,48);a.update(e[e['t']<20000],.02);b.update(e,.02)
    for key in ['quality','hz','mask']:np.testing.assert_array_equal(a.step(None,.02)[key],b.step(None,.02)[key])
    return dict(three_way_spectrum_exact_ticks=n,partial_oldest_chunk_gap_checked=True,expiration_empty=True,
        future_injection_ignored=True,scope='Numeric/statistics invariants, not target/RPM accuracy.')


if __name__=='__main__':print(run())
