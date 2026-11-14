import numpy as np
from windows21 import select,warp,Observer


def run():
    dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')]);e=np.zeros(6,dtype)
    e['x']=[50]*6;e['y']=40;e['t']=[79000,80000,90000,95000,99999,100000]
    assert select(e,.1,20000)['t'].tolist()==[80000,90000,95000,99999]
    assert select(e,.1,10000)['t'].tolist()==[90000,95000,99999]
    assert select(e,.1,5000)['t'].tolist()==[95000,99999]
    q=warp(select(e,.1,20000),.1,np.array([1000.,0.]),320,240)
    assert q['x'].tolist()==[70,60,55,50]
    np.testing.assert_array_equal(warp(select(e,.1,20000),.1,np.zeros(2),320,240),select(e,.1,20000))
    assert len(warp(select(e,.1,20000),.1,np.array([1e6,0.]),320,240))==1
    return dict(causal_window_boundaries=True,translation_shift_matches_analytic=True,zero_motion_exact=True,
        invalid_future_or_outside_points_not_emitted=True,scope='Geometry input invariants; native-window performance/accuracy measured separately.')


if __name__=='__main__':print(run())
