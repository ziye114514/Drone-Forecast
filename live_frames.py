"""Bounded shared native-frame pool with per-slot leases and generation checks.

Only this adapter owns image storage; event timestamps and SDK filtering are
unchanged. A descriptor can never silently refer to a later overwritten frame.
"""
from multiprocessing import shared_memory
import hashlib
import numpy as np


class SharedFrames:
    def __init__(self,context,height=720,width=1280,slots=3,preview=False,preview_window_ms=5.):
        self.owner=True;self.closed=False;self.preview=preview
        self.channels=2 if preview and preview_window_ms<20 else 1
        self.shape=(slots,self.channels,height,width,3)
        self.memory=shared_memory.SharedMemory(create=True,size=int(np.prod(self.shape)))
        self.images=np.ndarray(self.shape,np.uint8,buffer=self.memory.buf)
        self.locks=[context.Lock() for _ in range(slots)]
        self.generations=context.Array('q',slots,lock=False)
        self.sequence=0;self.next_slot=0

    def spec(self):
        return dict(name=self.memory.name,shape=self.shape,preview=self.preview)

    @classmethod
    def attach(cls,spec,locks,generations):
        self=object.__new__(cls);self.owner=False;self.closed=False
        self.memory=shared_memory.SharedMemory(name=spec['name'])
        self.shape=tuple(spec['shape']);self.preview=spec['preview'];self.channels=self.shape[1]
        self.images=np.ndarray(self.shape,np.uint8,buffer=self.memory.buf)
        self.locks=locks;self.generations=generations;self.sequence=0;self.next_slot=0
        return self

    def write_events(self,clean,t,preview_window_ms,verify=False):
        from metavision_sdk_core import BaseFrameGenerationAlgorithm
        selected=None
        for offset in range(self.shape[0]):
            index=(self.next_slot+offset)%self.shape[0]
            if self.locks[index].acquire(False):selected=index;break
        if selected is None:return None
        try:
            BaseFrameGenerationAlgorithm.generate_frame(clean,self.images[selected,0])
            if self.channels==2:
                recent=clean[clean['t']>=round(t*1e6-preview_window_ms*1000)]
                BaseFrameGenerationAlgorithm.generate_frame(recent,self.images[selected,1])
            self.sequence+=1;self.generations[selected]=self.sequence
            descriptor=dict(slot=selected,generation=self.sequence,
                detector_sha256=hashlib.sha256(self.images[selected,0].tobytes()).hexdigest() if verify else None,
                preview_sha256=hashlib.sha256(self.images[selected,-1].tobytes()).hexdigest() if verify and self.preview else None)
            self.next_slot=(selected+1)%self.shape[0]
            return descriptor
        finally:self.locks[selected].release()

    def read(self,descriptor,destination=None):
        index=descriptor['slot']
        with self.locks[index]:
            if self.generations[index]!=descriptor['generation']:return None
            if destination is None:frame=self.images[index,0].copy()
            else:
                if destination.shape!=self.shape[2:] or destination.dtype!=np.uint8:
                    raise ValueError('Shared detector destination shape/dtype mismatch')
                np.copyto(destination,self.images[index,0]);frame=destination
            preview=self.images[index,-1].copy() if self.preview else None
        return frame,preview

    def close(self):
        if self.closed:return
        self.closed=True;self.images=None;self.memory.close()
        if self.owner:self.memory.unlink()
