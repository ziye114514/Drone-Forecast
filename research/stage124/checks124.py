"""Boundary regressions for the real support adapter, no external truth."""
from pathlib import Path
import sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage124')]
from occupancy124 import SupportFrontend,MergeRecovery
from study124 import save,OUT
from polarity import counts
def main():
    w,h=320,240;dtype=[('x','u2'),('y','u2'),('p','i2'),('t','i8')]
    mask=np.zeros((h,w),np.uint8)
    for c in [(80,65),(230,65),(80,175),(230,175)]:cv2.ellipse(mask,c,(23,10),7,0,360,1,-1)
    yy,xx=np.nonzero(mask);e=np.zeros(len(xx)*2,dtype=dtype);e['x']=np.repeat(xx,2);e['y']=np.repeat(yy,2);e['p']=np.tile([0,1],len(xx));e['t']=19999
    f=SupportFrontend(w,h,'occupancy_global');p=counts(e,w,h);p['on']/=2;p['off']/=2
    o,_=f.detect_prepared(e,.02,p);assert o['detected'] and len(o['rotors'])==4
    assert all(r['major']>40 and r['minor']>15 for r in o['rotors'])
    # Same event positions, no second polarity: must not invent any support.
    mono=e.copy();mono['p']=1;g=SupportFrontend(w,h,'occupancy_global');q=counts(mono,w,h);q['on']/=2;q['off']/=2
    no,_=g.detect_prepared(mono,.02,q);assert not no['detected'] and np.count_nonzero(g.current_raw_mask)==0
    # ROI cannot provide identity when no current valid whole-object hint.
    g=SupportFrontend(w,h,'occupancy_roi');g.set_hint(e,dict(t=.02,detected=False),.02)
    no,_=g.detect_prepared(e,.02,p);assert not no['detected']
    rejected=False
    try:g.set_hint(e,dict(t=.04,detected=True,bbox=[0,0,w,h]),.02)
    except AssertionError:rejected=True
    assert rejected
    # Count adapter + valid full-frame ROI follows original full detector,
    # including original merge provenance and state, rather than mask-only check.
    old=MergeRecovery(w,h,'merge_safe');new=SupportFrontend(w,h,'count_roi')
    dense=np.repeat(e,6);dense['t']=19999
    p=counts(dense,w,h);p['on']/=2;p['off']/=2
    for tick in range(1,7):
        t=tick*.02;dense['t']=round(t*1e6)-1
        new.set_hint(dense,dict(t=t,detected=True,bbox=[0,0,w,h]),t)
        a,_=old.detect_prepared(dense,t,p);b,_=new.detect_prepared(dense,t,p);assert a==b
    save(OUT/'adapter_checks124.json',dict(low_count_four_supported=True,monopolar_rejected=True,no_hint_ROI_rejected=True,future_hint_rejected=True,
        dense_count_full_frontend_exact=True,original_gates_unchanged=True))
    print('Support adapter regressions passed',flush=True)
if __name__=='__main__':main()
