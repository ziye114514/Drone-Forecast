"""Invariants for mutable temporal state and publication semantics."""
import numpy as np
from frontend18 import IncrementalSpectrum,IncrementalFrontend,TrackPublisher,old

DTYPE=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')])


def run():
    rng=np.random.default_rng(5418);e=np.zeros(100000,DTYPE)
    e['x']=rng.integers(0,64,len(e));e['y']=rng.integers(0,48,len(e));e['p']=rng.integers(0,2,len(e))
    e['t']=np.sort(rng.integers(0,600000,len(e)))
    ring=IncrementalSpectrum(64,48);reference=old.SparseSpectrum(64,48);compared=0
    for hi in range(20000,600001,20000):
        current=e[(e['t']>=hi-20000)&(e['t']<hi)];ring.update(current,hi/1e6)
        recent=e[(e['t']>=hi-100000)&(e['t']<hi)]
        a=reference.step(recent,hi/1e6);b=ring.step(None,hi/1e6)
        for key in ['quality','hz','mask']:np.testing.assert_array_equal(a[key],b[key])
        for key in ['active_blocks','eligible_blocks','regions','events']:assert a[key]==b[key],key
        compared+=1
    empty=np.empty(0,DTYPE);ring.update(empty,.82);assert ring.step(None,.82)['events']==0
    try:ring.update(empty,.82)
    except ValueError:pass
    else:raise AssertionError('repeated cutoff accepted')
    # Same current slice plus future events has identical histogram/evidence.
    a=IncrementalSpectrum(64,48);b=IncrementalSpectrum(64,48)
    a.update(e[e['t']<20000],.02);b.update(e,.02)
    np.testing.assert_array_equal(a.ring,b.ring)
    # Preserve full geometric observations versus frozen frontend on monopolar4disks.
    import cv2
    mask=np.zeros((240,320),np.uint8)
    for c in [(90,70),(230,70),(90,170),(230,170)]:cv2.ellipse(mask,c,(22,7),12,0,360,1,-1)
    yy,xx=np.nonzero(mask);stamps=np.arange(1000,400000,1e6/300).astype('i8')
    events=np.zeros(len(xx)*len(stamps),DTYPE);events['x']=np.tile(xx,len(stamps));events['y']=np.tile(yy,len(stamps));events['p']=1;events['t']=np.repeat(stamps,len(xx))
    new=IncrementalFrontend(320,240);legacy=old.Frontend(320,240,'global');equal=0
    for hi in range(20000,400001,20000):
        current=events[(events['t']>=hi-20000)&(events['t']<hi)];ready=hi>=100000
        q=new.advance(current,hi/1e6,ready)
        if ready:
            expected=legacy.step(events[(events['t']>=hi-100000)&(events['t']<hi)],current,hi/1e6)
            assert q['observation']==expected['observation'];equal+=1
    def proposal(x=100):return dict(detected=True,center=[x,100],bbox=[x-50,70,x+50,130],rotors=[])
    gate=TrackPublisher()
    assert not gate.apply(proposal(),.10)['observation']['detected']
    assert not gate.apply(proposal(),.12)['observation']['detected']
    assert gate.apply(proposal(),.14)['observation']['detected']
    assert not gate.apply(None,.16)['observation']['detected']
    assert gate.apply(proposal(),.18)['observation']['detected']
    assert not gate.apply(proposal(300),.20)['observation']['detected']
    assert gate.apply(proposal(),.22)['observation']['detected']
    assert not gate.apply(None,.40)['observation']['detected']
    assert not gate.apply(proposal(),.42)['observation']['detected']
    return dict(ring_reference_exact_ticks=compared,full_geometry_exact_synthetic_ticks=equal,
        ring_wrap_and_expiration=True,cutoff_guard=True,future_injection_ignored=True,
        initial3_required=True,gap_can_resume_measurement=True,no_coast_publication=True,
        outlier_rejected=True,expired_track_reinitializes=True,
        limitation='Synthetic invariants; not real-world accuracy evidence.')


if __name__=='__main__':print(run())
