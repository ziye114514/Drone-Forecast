"""Exact deterministic reservoir histogram for chronologically ordered EventCD."""
import ctypes as C
from pathlib import Path
import numpy as np

class NativeTimingHistogram:
    def __init__(self,path):
        self.library=C.CDLL(str(Path(path).resolve()));self.handles={}
        self.create=self.library.create_timing_histogram;self.create.argtypes=[C.c_int]*5;self.create.restype=C.c_void_p
        self.destroy=self.library.destroy_timing_histogram;self.destroy.argtypes=[C.c_void_p]
        self.process=self.library.timing_histogram
        self.process.argtypes=[C.c_void_p,C.c_void_p,C.c_int64,C.c_int64]+[C.c_int]*4+[C.c_int64]*2+[C.c_void_p]*2;self.process.restype=C.c_int

    def __call__(self,events,begin,end,width,height,cell,bins,block):
        fields=events.dtype.fields
        for name,size in [('x',2),('y',2),('p',2),('t',8)]:
            if fields[name][0].itemsize!=size:raise ValueError('Native histogram expects standard EventCD types')
        key=(width,height,cell,bins,block)
        if key not in self.handles:
            handle=self.create(*key)
            if not handle:raise MemoryError('Native timing workspace allocation failed')
            self.handles[key]=handle
        groups=((width+cell-1)//cell)*((height+cell-1)//cell)*2
        histogram=np.empty((groups,2*bins),np.float32);spatial=np.empty(groups,np.int64)
        code=self.process(self.handles[key],events.ctypes.data,len(events),events.strides[0],
            *[fields[n][1] for n in ('x','y','p','t')],begin,end,histogram.ctypes.data,spatial.ctypes.data)
        if code:raise ValueError(f'Native histogram error{code}: ordered timestamps and valid native XY required')
        return histogram,spatial

    def close(self):
        for handle in self.handles.values():self.destroy(handle)
        self.handles.clear()
