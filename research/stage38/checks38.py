import numpy as np
from common38 import PARAMS,source_box,viewport
from render38 import project,activity_filter,render


def run():
    e=np.zeros(6,dtype=np.dtype([('x','<u2'),('y','<u2'),('p','<i2'),('t','<i8')],align=True))
    e['x']=[2,3,10,2,3,9];e['y']=[2,2,8,3,3,8];e['p']=[0,1,0,1,0,1];e['t']=[10001,10002,10003,10004,10005,30000]
    filtered=activity_filter(e,16,16);prefix=activity_filter(e[:4],16,16)
    for field in ['x','y','p','t']:assert np.array_equal(prefix[field],filtered[filtered['t']<=10004][field])
    assert len(filtered)>0
    for d in PARAMS['divisors']:
        q,v=project(e,16,16,d)
        assert np.array_equal(q['t'],e['t']) and np.array_equal(q['p'],e['p'])
        restored_x=(q['x'].astype(float)-v[0])*d;restored_y=(q['y'].astype(float)-v[1])*d
        assert np.all((e['x']>=restored_x)&(e['x']<restored_x+d)) and np.all((e['y']>=restored_y)&(e['y']<restored_y+d))
        b=[v[0]+.2,v[1]+.2,v[0]+16/d-.2,v[1]+16/d-.2];mapped=source_box(b,v,d,16,16);assert np.allclose(mapped,[.2*d,.2*d,16-.2*d,16-.2*d])
        assert render(e,16,16,d)[0].shape==(16,16,3)
    assert source_box([0,0,1,1],viewport(1280,720,8),8,1280,720) is None
    return dict(activity_filter_causal_prefix=True,projection_timestamp_polarity_unchanged=True,bounded_pixel_quantization=True,
        bbox_inverse_exact_continuous_transform=True,padding_center_rejected=True,native_sdk_palette=True)
