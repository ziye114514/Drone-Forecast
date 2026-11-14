"""Streaming SDK activity filtering, retaining native coordinates and event polarity."""
import numpy as np
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm

PARAMS=dict(threshold_us=10000,step_us=10000,filter='Installed SDK ActivityNoiseFilterAlgorithm',
    scope='One persistent filter per RAW segment, no spatial rescale/image reconstruction. Empty current chunk stays empty; copy accepted events before reusable SDK buffer changes.')


class StreamFilter:
    def __init__(self,w,h):self.algorithm=ActivityNoiseFilterAlgorithm(w,h,PARAMS['threshold_us']);self.buffer=self.algorithm.get_empty_output_buffer()

    def step(self,events):
        if not len(events):return events.copy()
        self.algorithm.process_events(events,self.buffer);return self.buffer.numpy().copy()


def checks():
    e=np.zeros(7,dtype=np.dtype([('x','<u2'),('y','<u2'),('p','<i2'),('t','<i8')],align=True));e['x']=[2,3,12,3,2,3,12];e['y']=[2,2,12,3,2,3,12];e['p']=[0,1,0,1,0,1,0];e['t']=[10001,10002,10003,10004,20000,20001,40000]
    whole=StreamFilter(16,16).step(e);chunked=StreamFilter(16,16);a=chunked.step(e[:3]);saved=a.copy();b=chunked.step(e[3:]);joined=np.concatenate([a,b])
    assert np.array_equal(a,saved)
    for field in ['x','y','p','t']:assert np.array_equal(whole[field],joined[field])
    assert len(chunked.step(e[:0]))==0
    future=StreamFilter(16,16).step(e[:4]);assert np.array_equal(future['t'],whole[whole['t']<=10004]['t'])
    return dict(sdk_streaming_chunking_exact=True,output_copy_survives_buffer_reuse=True,empty_current_chunk_no_stale_events=True,causal_prefix_exact=True)
