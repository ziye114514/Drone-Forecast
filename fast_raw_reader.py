"""Bounded two-slot RAW decoder process; native bytes, no frame dropping."""
from multiprocessing import shared_memory
import multiprocessing as mp
import time
import traceback


def produce(raw,start,duration,step_us,names,capacity,ready,free,filtered_names=None,read_chunk=100000,avx2_filter=False,read_buffers=3,reader_backend='iterator',defer_cyclic_gc=False):
    import numpy as np
    from live_decode import native_batches
    import gc
    gc_events=[];gc_started={};reported_gc=0;gc_enabled=gc.isenabled()
    def gc_callback(phase,info):
        generation=info['generation']
        if phase=='start':gc_started[generation]=time.perf_counter()
        elif generation in gc_started:
            gc_events.append(dict(generation=generation,duration_ms=(time.perf_counter()-gc_started.pop(generation))*1000,
                collected=info['collected']))
    gc.callbacks.append(gc_callback)
    if defer_cyclic_gc:gc.collect();gc_events.clear();gc.disable()
    buffers=[shared_memory.SharedMemory(name=n,create=False) for n in names]
    filtered=[]
    try:
        if filtered_names:
            if avx2_filter:
                from sdk_activity_backend import SDKActivityAVX2 as ActivityNoiseFilterAlgorithm
            else:from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
            filtered=[shared_memory.SharedMemory(name=n,create=False) for n in filtered_names]
            noise=ActivityNoiseFilterAlgorithm(1280,720,10000);clean_buffer=noise.get_empty_output_buffer()
        iterator=iter(native_batches(raw,start,duration,False,step_us,read_chunk=read_chunk,read_buffers=read_buffers,reader_backend=reader_backend))
        while True:
            slot=free.get();begin=time.perf_counter();cpu_begin=time.process_time();packet=next(iterator,None)
            decoded=time.perf_counter()
            decode_cpu=(time.process_time()-cpu_begin)*1000
            if packet is None:ready.put(None);break
            t,e=packet
            from metavision_sdk_base import EventCD
            if e.dtype!=np.dtype(EventCD):
                raise TypeError('Native event dtype does not match16-byte SDK EventCD; refuse shared-memory reinterpretation')
            if not e.flags.c_contiguous:e=np.ascontiguousarray(e)
            filtered_count=None
            if filtered:
                noise.process_events(e,clean_buffer);clean=clean_buffer.numpy();filtered_count=len(clean)
                if not clean.flags.c_contiguous:clean=np.ascontiguousarray(clean)
            filtered_done=time.perf_counter()
            component=dict(native_decode_ms=(decoded-begin)*1000,activity_filter_ms=(filtered_done-decoded)*1000,
                native_decode_CPU_ms=decode_cpu,
                native_decode_wall_minus_CPU_ms=max(0.,(decoded-begin)*1000-decode_cpu),
                filter_provenance=None if not filtered else getattr(noise,'provenance',dict(backend='installed_SDK_Python_binding')))
            component['cyclic_GC_events']=gc_events[reported_gc:];reported_gc=len(gc_events)
            overflow=bool(e.nbytes>capacity or (filtered and clean.nbytes>capacity))
            if overflow:
                # Rare sensor bursts exceed a nominal shared slot. Preserve all
                # native bytes in an immutable queued packet; queue length is
                # still bounded2. No frame/event truncation or time retiming.
                raw_bytes=e.tobytes();clean_bytes=clean.tobytes() if filtered else None
                free.put(slot)
                ready.put(dict(t=t,n=len(e),slot=None,filtered_count=filtered_count,
                    raw_bytes=raw_bytes,filtered_bytes=clean_bytes,overflow=True,
                    decode_ms=(time.perf_counter()-begin)*1000,**component))
                continue
            target=np.ndarray((e.nbytes,),np.uint8,buffer=buffers[slot].buf)
            np.copyto(target,e.view(np.uint8));target=None
            if filtered:
                target=np.ndarray((clean.nbytes,),np.uint8,buffer=filtered[slot].buf)
                np.copyto(target,clean.view(np.uint8));target=None
            ready.put(dict(t=t,n=len(e),slot=slot,filtered_count=filtered_count,decode_ms=(time.perf_counter()-begin)*1000,**component))
    except BaseException:
        ready.put(dict(error=traceback.format_exc()))
    finally:
        gc.callbacks.remove(gc_callback)
        if gc_enabled:gc.enable()
        for b in buffers+filtered:b.close()


class ProcessRAW:
    def __init__(self,raw,start,duration,step_us,filter_events=False,read_chunk=100000,avx2_filter=False,read_buffers=3,reader_backend='iterator',slot_events=2000000,defer_cyclic_gc=False):
        import numpy as np
        from metavision_sdk_base import EventCD
        if not isinstance(slot_events,int) or not 1<=slot_events<=16000000:
            raise ValueError('slot_events must be an integer in1..16000000')
        self.np=np;self.dtype=np.dtype(EventCD);self.slot_events=slot_events;self.capacity=slot_events*self.dtype.itemsize
        self.buffers=[shared_memory.SharedMemory(create=True,size=self.capacity) for _ in range(2)]
        self.filtered_buffers=[shared_memory.SharedMemory(create=True,size=self.capacity) for _ in range(2)] if filter_events else []
        self.allocated_shared_bytes=self.capacity*(len(self.buffers)+len(self.filtered_buffers))
        context=mp.get_context('spawn');self.ready=context.Queue(maxsize=2);self.free=context.Queue(maxsize=2)
        for i in range(2):self.free.put(i)
        self.previous=None;self.view=None;self.filtered_events=None;self.overflow_packets=0;self.max_packet_events=0;self.packet_overflow=False
        self.filter_provenance=None;self.last_native_decode_ms=0.;self.last_activity_filter_ms=0.
        self.last_native_decode_CPU_ms=0.;self.last_native_decode_wait_proxy_ms=0.
        self.gc_events=[];self.last_gc_events=[];self.defer_cyclic_gc=defer_cyclic_gc
        self.process=context.Process(target=produce,args=(str(raw),start,duration,step_us,
            [b.name for b in self.buffers],self.capacity,self.ready,self.free,
            [b.name for b in self.filtered_buffers],read_chunk,avx2_filter,read_buffers,reader_backend,defer_cyclic_gc),daemon=True)
        self.process.start()
    def read(self):
        self.view=None;self.filtered_events=None
        if self.previous is not None:self.free.put(self.previous);self.previous=None
        packet=self.ready.get(timeout=120)
        if packet is None:return None
        if 'error' in packet:raise RuntimeError(packet['error'])
        self.filter_provenance=packet.get('filter_provenance')
        self.last_native_decode_ms=packet.get('native_decode_ms',0.);self.last_activity_filter_ms=packet.get('activity_filter_ms',0.)
        self.last_native_decode_CPU_ms=packet.get('native_decode_CPU_ms',0.)
        self.last_native_decode_wait_proxy_ms=packet.get('native_decode_wall_minus_CPU_ms',0.)
        self.last_gc_events=packet.get('cyclic_GC_events',[]);self.gc_events.extend(self.last_gc_events)
        self.packet_overflow=bool(packet.get('overflow'))
        self.max_packet_events=max(self.max_packet_events,packet['n'])
        if packet.get('overflow'):
            self.overflow_packets+=1
            self.view=self.np.frombuffer(packet['raw_bytes'],dtype=self.dtype)
            if self.filtered_buffers:self.filtered_events=self.np.frombuffer(packet['filtered_bytes'],dtype=self.dtype)
            return packet['t'],self.view,packet['decode_ms']
        self.previous=packet['slot']
        self.view=self.np.ndarray((packet['n'],),self.dtype,buffer=self.buffers[self.previous].buf)
        self.view.setflags(write=False)
        if self.filtered_buffers:
            self.filtered_events=self.np.ndarray((packet['filtered_count'],),self.dtype,buffer=self.filtered_buffers[self.previous].buf)
            self.filtered_events.setflags(write=False)
        return packet['t'],self.view,packet['decode_ms']
    def close(self):
        self.view=None;self.filtered_events=None
        if self.previous is not None:
            try:self.free.put_nowait(self.previous)
            except Exception:pass
            self.previous=None
        self.process.join(timeout=3)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=5)
        for q in (self.ready,self.free):q.cancel_join_thread();q.close()
        for b in self.buffers+self.filtered_buffers:b.close();b.unlink()
