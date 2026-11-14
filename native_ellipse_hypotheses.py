"""Explicit optional batch AMS backend; no silent replacement/fallback."""
from pathlib import Path
import ctypes as C
import os
import numpy as np

class NativeEllipseHypotheses:
    def __init__(self,dll_path):
        sdk=Path('D:/Prophesee/third_party/bin')
        self.dll_directory=os.add_dll_directory(str(sdk))
        self.library=C.CDLL(str(Path(dll_path).resolve()))
        self.function=self.library.batch_ellipse_ams
        self.function.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_void_p];self.function.restype=C.c_int

    def __call__(self,points,indices):
        samples=np.ascontiguousarray(np.asarray(points)[indices],dtype=np.float32)
        n,k,_=samples.shape;out=np.empty((n,5),np.float64);good=np.empty(n,np.int32)
        status=self.function(samples.ctypes.data,n,k,out.ctypes.data,good.ctypes.data)
        if status:raise RuntimeError(f'Native AMS returned {status}')
        return [p.copy() if keep else None for p,keep in zip(out,good)]
