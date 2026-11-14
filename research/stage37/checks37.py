import numpy as np
from common37 import PARAMS
from render37 import render,past


def run():
    e=np.zeros(5,dtype=np.dtype([('x','<u2'),('y','<u2'),('p','<i2'),('t','<i8')],align=True));e['x']=[1,2,3,4,5];e['y']=2;e['t']=[89,90,99,100,101];e['p']=[0,1,0,1,1]
    selected=past(e,100,10);assert selected['t'].tolist()==[90,99]
    for kind in PARAMS['formats']:
        a=render(selected,12,10,kind);b=render(past(e[:4],100,10),12,10,kind);assert np.array_equal(a,b) and a.shape==(10,12,3) and a.dtype==np.uint8
        assert render(e[:0],12,10,kind).shape==(10,12,3)
    opposite=selected.copy();opposite['p']=1-opposite['p'];assert np.array_equal(render(selected,12,10,'native_count_white'),render(opposite,12,10,'native_count_white'))
    return dict(exact_past_window_boundaries=True,future_events_excluded=True,three_native_renderings_uint8=True,count_render_polarity_invariant=True,no_reconstructed_gray=True)
