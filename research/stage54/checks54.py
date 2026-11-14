import copy,sys
import numpy as np
from packets54 import *

def raises(kind,fn):
    try:fn()
    except kind:return
    raise AssertionError('Expected '+kind.__name__)
def run():
    dtype=np.dtype(dict(names=['x','y','p','t'],formats=['<u2','<u2','u1','<i8'],offsets=[0,2,4,8],itemsize=16),align=True)
    events=np.zeros(4,dtype);events['x']=[10,11,12,13];events['y']=20;events['p']=[0,1,0,1];events['t']=[0,9999,10000,19999]
    assert restore_dtype(dtype_spec(dtype))==dtype
    p=Publisher(4096);name=p.memory.name;r=PacketReader(name,1280,720,0.);stream=r.chunks()
    try:
        header=p.publish(events,0,.02,True);raises(AssertionError,lambda:p.publish(events,1,.02,True))
        r.set_request(dict(id=0,cutoff=.02,packet=header));assert not r.data.flags.writeable
        t,a=next(stream);u,b=next(stream);assert (t,u)==(.01,.02)
        np.testing.assert_array_equal(np.concatenate([a,b]),events);assert a['t'].tolist()==[0,9999] and b['t'].tolist()==[10000,19999]
        p.acknowledge(0)
        header=p.publish(events[:0],1,.04,True);r.set_request(dict(id=1,cutoff=.04,packet=header));assert next(stream)[1].size==0 and next(stream)[1].size==0;p.acknowledge(1)
        bad=events.copy();bad['t'][-1]=60000;raises(AssertionError,lambda:p.publish(bad,2,.06,True))
        valid=events.copy();valid['t']+=40000;header=p.publish(valid,2,.06,True);wrong=copy.deepcopy(header);wrong['payload_sha256']='0'*64
        raises(AssertionError,lambda:r.set_request(dict(id=2,cutoff=.06,packet=wrong)))
        r.set_request(dict(id=2,cutoff=.06,packet=header));next(stream);next(stream);p.acknowledge(2)
        raises(AssertionError,lambda:r.set_request(dict(id=2,cutoff=.06,packet=header)))
        obj=np.dtype([('x',object),('y','i4'),('p','i1'),('t','i8')]);raises(AssertionError,lambda:dtype_spec(obj))
    finally:stream.close();r.close();p.close()
    try:
        leftover=shared_memory.SharedMemory(name=name,create=False)
    except FileNotFoundError:pass
    else:leftover.close();raise AssertionError('Owned shared memory leaked')
    p=Publisher(16)
    try:raises(BufferError,lambda:p.publish(events,0,.02,False))
    finally:p.close()
    from pathlib import Path
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root));import run_deploy_forecast as entry
    from rotor_worker54 import worker_source
    compile(worker_source(),'<worker control>','exec')
    for mode in ['dual','shared']:
        namespace=entry.build(mode,32)
        try:assert namespace['METHODS']==['delayed_feedback'] and len(namespace['MODELS'])==15
        finally:namespace.clear()
    return dict(padded_dtype_offsets_and_polarity_preserved=True,two_native10_boundaries_exact=True,empty_packet_supported=True,
        input_order_no_sampling=True,checksum_corruption_and_stale_id_rejected=True,future_event_rejected=True,
        backpressure_before_ack=True,oversize_raises_without_dropping=True,object_dtype_rejected=True,owned_memory_released=True,
        both_modes_keep_complete15_models=True)
