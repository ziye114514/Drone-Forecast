"""Optional compiled AMS seed bank and distance arithmetic, original gates."""
import ctypes as C
import numpy as np
from native_ellipse_hypotheses import NativeEllipseHypotheses

class NativeGeometryKernels(NativeEllipseHypotheses):
    def __init__(self,path):
        super().__init__(path)
        self.distance_function=self.library.ellipse_distance_batch
        self.distance_function.argtypes=[C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_void_p];self.distance_function.restype=C.c_int

    def distances_batch(self,points,parameters):
        pts=np.ascontiguousarray(points,dtype=np.float64).reshape(-1,2);p=np.ascontiguousarray(parameters,dtype=np.float64).reshape(-1,5)
        out=np.empty((len(p),len(pts)),np.float64)
        code=self.distance_function(pts.ctypes.data,len(pts),p.ctypes.data,len(p),out.ctypes.data)
        if code:raise RuntimeError(f'Native conic distance error {code}')
        return out

    def distance(self,points,p):return self.distances_batch(points,p)[0]

def load_backend(path):
    """Select only capabilities actually exported by an explicit local DLL."""
    basic=NativeGeometryKernels(path)
    if hasattr(basic.library,'arc_search215'):
        from native_arc_search import NativeArcSearch
        return NativeArcSearch(path)
    return basic
