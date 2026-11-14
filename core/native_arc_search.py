"""Optional compiled full hypothesis/refinement loop; Python uncertainty stays."""
import ctypes as C
import numpy as np
from native_geometry_kernels import NativeGeometryKernels
from research.stage147.curvature147 import PARAMS

class NativeArcSearch(NativeGeometryKernels):
    def __init__(self,path):
        super().__init__(path);self.search_function=self.library.arc_search215
        self.search_function.argtypes=[C.c_void_p,C.c_int,C.c_double,C.c_void_p,C.c_int,
            C.c_double,C.c_double,C.c_void_p,C.c_void_p];self.search_function.restype=C.c_int
        if hasattr(self.library,'arc_uncertainty216'):
            self.uncertainty_function=self.library.arc_uncertainty216
            self.uncertainty_function.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_void_p]
            self.uncertainty_function.restype=C.c_int
            self.uncertainty=self._native_uncertainty

    def _native_uncertainty(self,points,parameters):
        p=np.ascontiguousarray(points,dtype=np.float64).reshape(-1,2)
        q=np.ascontiguousarray(parameters,dtype=np.float64);out=np.empty(3,np.float64)
        status=self.uncertainty_function(p.ctypes.data,len(p),q.ctypes.data,out.ctypes.data)
        if status:raise RuntimeError(f'Native uncertainty error {status}')
        return tuple(float(x) for x in out)

    def search(self,points,extent,indices):
        p=np.ascontiguousarray(points,dtype=np.float64);ix=np.ascontiguousarray(indices,dtype=np.int32)
        out=np.empty(8,np.float64);keep=np.empty(len(p),np.uint8)
        status=self.search_function(p.ctypes.data,len(p),extent,ix.ctypes.data,len(ix),
            PARAMS['minimum_arc_coverage'],PARAMS['maximum_arc_rms_px'],out.ctypes.data,keep.ctypes.data)
        if status<0:raise RuntimeError(f'Native arc search error {status}')
        if not status:return None
        return out[:5].copy(),p[keep>0],float(out[5]),float(out[6]),float(out[7])
