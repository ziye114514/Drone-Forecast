"""Fuse the unchanged Sampson residual arithmetic, preserving every point."""
import time
import numpy as np
from native_counts_numba import njit


@njit(cache=True,nogil=True)
def residuals(points,p,cosine,sine):
    values=np.empty(len(points));a=p[2];b=p[3];a2=a*a;b2=b*b
    for i in range(len(points)):
        dx=points[i,0]-p[0];dy=points[i,1]-p[1]
        x=dx*cosine+dy*sine;y=-dx*sine+dy*cosine
        xa=x/a2;yb=y/b2;gradient=2*np.sqrt(xa*xa+yb*yb)
        X=x/a;Y=y/b
        values[i]=(X*X+Y*Y-1)/max(gradient,1e-6)
    return values


class ConicResidual:
    def __init__(self):
        start=time.perf_counter();points=np.empty((0,2));p=np.array([0.,0.,10.,3.,.2])
        residuals(points,p,np.cos(p[4]),np.sin(p[4]))
        self.calls=0
        self.provenance=dict(warm_compile_load_ms=(time.perf_counter()-start)*1000,
            same_Sampson_residual=True,points_sampled=False,fastmath=False,
            NumPy_scalar_trigonometry=True,numerical_roundoff_not_bitwise_claim=True)

    def __call__(self,points,p):
        self.calls+=1
        return residuals(np.ascontiguousarray(points,dtype=np.float64),p,np.cos(p[4]),np.sin(p[4]))
