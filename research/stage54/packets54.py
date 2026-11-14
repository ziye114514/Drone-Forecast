"""Bounded, acknowledged native structured-event packets; no sampling or reorder."""
from multiprocessing import shared_memory
import hashlib,time
import numpy as np

def dtype_spec(dtype):
    assert dtype.names and not dtype.hasobject and {'x','y','p','t'}<=set(dtype.names)
    for name in dtype.names:assert dtype.fields[name][0].subdtype is None and dtype.fields[name][0].kind in 'biuf'
    return dict(names=list(dtype.names),formats=[dtype.fields[n][0].str for n in dtype.names],offsets=[int(dtype.fields[n][1]) for n in dtype.names],itemsize=int(dtype.itemsize),aligned=bool(dtype.isalignedstruct))
def restore_dtype(spec):
    result=np.dtype({k:spec[k] for k in ['names','formats','offsets','itemsize']},align=bool(spec['aligned']))
    assert dtype_spec(result)==spec;return result

class Publisher:
    def __init__(self,capacity_bytes):self.memory=shared_memory.SharedMemory(create=True,size=capacity_bytes);self.pending=None;self.next_id=0;self.max_bytes=0
    def publish(self,events,request_id,cutoff,verify):
        tick=time.perf_counter();assert self.pending is None and request_id==self.next_id
        events=np.ascontiguousarray(events);spec=dtype_spec(events.dtype);end=round(cutoff*1e6);start=end-20000
        assert end%20000==0 and (not len(events) or (int(events['t'].min())>=start and int(events['t'].max())<end))
        nbytes=events.nbytes
        if nbytes>self.memory.size:raise BufferError(f'Native event packet needs {nbytes} bytes, capacity {self.memory.size}; no events dropped.')
        data=memoryview(events).cast('B')
        try:self.memory.buf[:nbytes]=data;digest=hashlib.sha256(data).hexdigest() if verify else None
        finally:data.release()
        self.pending=request_id;self.next_id+=1;self.max_bytes=max(self.max_bytes,nbytes)
        return dict(count=len(events),dtype=spec,nbytes=nbytes,start_us=start,end_us=end,payload_sha256=digest,publish_ms=(time.perf_counter()-tick)*1000)
    def acknowledge(self,request_id):assert self.pending==request_id;self.pending=None
    def close(self):self.memory.close();self.memory.unlink()

class PacketReader:
    def __init__(self,name,width,height,start):
        self.memory=shared_memory.SharedMemory(name=name,create=False);self.width=width;self.height=height;self.last_us=round(start*1e6);self.next_id=0;self.data=None;self.active=None;self.payload_sha256=None
    def get_size(self):return self.height,self.width
    def is_done(self):return False
    def set_request(self,request):
        assert request['id']==self.next_id;header=request['packet'];assert header['start_us']==self.last_us and header['end_us']-self.last_us==20000
        assert header['end_us']==round(float(request['cutoff'])*1e6)
        dtype=restore_dtype(header['dtype']);count=int(header['count']);assert count>=0 and count*dtype.itemsize==header['nbytes']<=self.memory.size
        data=np.ndarray((count,),dtype=dtype,buffer=self.memory.buf);data.setflags(write=False)
        try:
            assert not len(data) or int(data['t'].min())>=self.last_us and int(data['t'].max())<header['end_us']
            payload=memoryview(data).cast('B')
            try:digest=hashlib.sha256(payload).hexdigest() if header['payload_sha256'] is not None else None
            finally:payload.release()
            assert digest==header['payload_sha256']
        except BaseException:data=None;raise
        self.data=data;self.active=header;self.payload_sha256=digest;self.next_id+=1
    def chunks(self):
        data=None
        try:
            while True:
                assert self.active is not None and self.data is not None
                header=self.active;data=self.data
                for end in [header['start_us']+10000,header['end_us']]:
                    events=data[(data['t']>=self.last_us)&(data['t']<end)]
                    self.last_us=end
                    yield end/1e6,events
                data=None
        finally:data=None
    def close(self):self.data=None;self.active=None;self.memory.close()
