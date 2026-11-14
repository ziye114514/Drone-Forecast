"""Small dense convex Huber solver, reused established Newton/Armijo method."""
import numpy as np
from native_counts_numba import njit


@njit(cache=True,nogil=True)
def loss(residual,delta):
    total=0.
    for value in residual:
        v=abs(value)
        total+=.5*v*v if v<=delta else delta*(v-.5*delta)
    return total


@njit(cache=True,nogil=True)
def solve(J,target,initial,delta):
    p=initial.copy();n=len(p);evaluations=0
    for iteration in range(30):
        residual=J@p-target;gradient=np.zeros(n);H=np.zeros((n,n))
        for i in range(len(target)):
            clipped=min(delta,max(-delta,residual[i]))
            for a in range(n):
                gradient[a]+=J[i,a]*clipped
                if abs(residual[i])<=delta:
                    for b in range(n):H[a,b]+=J[i,a]*J[i,b]
        g=float(np.max(np.abs(gradient)))
        if g<=1e-8:return p,True,evaluations+1,g
        regularizer=max(1e-12,float(np.max(np.diag(H)))*1e-12)
        for a in range(n):H[a,a]+=regularizer
        direction=-np.linalg.solve(H,gradient)
        descent=float(gradient@direction)
        if not np.isfinite(direction).all() or descent>=0.:
            return p,False,evaluations+1,g
        before=loss(residual,delta);alpha=1.;accepted=False
        for search in range(40):
            candidate=p+alpha*direction;after=loss(J@candidate-target,delta);evaluations+=1
            if after<=before+1e-4*alpha*descent:
                p=candidate;accepted=True;break
            alpha*=.5
        if not accepted:return p,False,evaluations,g
    residual=J@p-target;gradient=J.T@np.minimum(delta,np.maximum(-delta,residual))
    g=float(np.max(np.abs(gradient)))
    return p,g<=1e-8,evaluations,g
