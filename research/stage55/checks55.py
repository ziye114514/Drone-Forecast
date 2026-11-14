import copy,sys
import numpy as np
from roi55 import *

def raises(fn):
    try:fn()
    except AssertionError:return
    raise AssertionError('Future hint should be rejected')
def run():
    dtype=np.dtype(dict(names=['x','y','p','t'],formats=['<u2','<u2','<i2','<i8'],offsets=[0,2,4,8],itemsize=16))
    e=np.zeros(4,dtype);e['x']=[2,8,20,40];e['y']=20;e['p']=[0,1,0,1];e['t']=[0,1,2,3]
    hint=dict(t=1.,detected=True,bbox=[10,10,30,30],center=[20,20]);on=GeometryROI(100,80,'body');off=GeometryROI(100,80,'off')
    cropped,d=on.apply(e,hint,1.01);np.testing.assert_array_equal(cropped,e[[1,2]]);assert d['roi']==[5,5,35,35]
    assert off.apply(e,hint,1.01)[0] is e and on.apply(e,None,1.01)[0] is e and on.apply(e,hint,1.17)[0] is e
    assert on.apply(e,hint,1.16)[1]['mode']=='body_roi';raises(lambda:on.apply(e,hint,.99))
    request=dict(packet=dict(start_us=1000000),body_hint=hint);validate_hint_request(request)
    future=copy.deepcopy(request);future['body_hint']['t']=1.01;raises(lambda:validate_hint_request(future))
    # Different opaque bytes, identical events: canonical fields SHA must agree.
    raw=bytearray(e.tobytes());bytes_view=np.frombuffer(raw,np.uint8).reshape(-1,16);bytes_view[:,[6,7]]=165
    padded=np.frombuffer(raw,dtype);assert named_fields_sha(e)==named_fields_sha(padded)
    changed=e.copy();changed['p'][0]=1;assert named_fields_sha(changed)!=named_fields_sha(e)
    assert named_fields_sha(e[::-1])!=named_fields_sha(e)
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root));import run_roi_forecast as entry
    from rotor_worker55 import worker_source
    compile(worker_source(),'<55 worker>','exec')
    for mode in ['off','body']:
        namespace=entry.build(mode,32)
        try:assert namespace['METHODS']==['delayed_feedback'] and len(namespace['MODELS'])==15
        finally:namespace.clear()
    return dict(old_inside17_executed=True,off_and_missing_stale_hint_preserve_original_event_object=True,
        roi_crops_only_spatial_current_values=True,original_polarity_timestamps_order_preserved=True,
        hint_must_be_known_by_packet_start=True,earlier10ms_cannot_use_future20ms_hint=True,
        canonical_hash_ignores_only_padding_and_detects_field_or_order_change=True,complete15_models_retained=True)
