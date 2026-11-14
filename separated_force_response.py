"""Fast unknown normal amplitude, smooth external disturbance rate.

Same two-force output a=k*u+d, with d=e+h*n. External e has an OU rate j;
hidden normal response h can change without giving wind arbitrary fast2D noise.
This is a causal image-response prior, not identifiable mechanical forces.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm
from normal_response_observer import NormalResponseObserver

RATE_TAU=.35
GAUSS_NODES,GAUSS_WEIGHTS=np.polynomial.legendre.leggauss(8)

@lru_cache(maxsize=64)
def blocks(dt_us,decay):
    dt=dt_us*1e-6
    A=np.zeros((4,4));A[0,1]=A[1,2]=A[2,3]=1.;A[2,2]=-decay;A[3,3]=-1/RATE_TAU
    def noise(index,spectral):
        G=np.zeros((4,4));G[index,index]=spectral
        M=np.zeros((8,8));M[:4,:4]=-A;M[:4,4:]=G;M[4:,4:]=A.T
        E=expm(M*dt);F=E[4:,4:].T;Q=F@E[:4,4:]
        return F,(Q+Q.T)/2
    F,Qj=noise(3,2*(1000./RATE_TAU)**2/RATE_TAU)
    _,Qh=noise(2,2*1000.**2/RATE_TAU)
    # Cross covariance of the integrated h noise with its scalar amplitude.
    nodes,weights=GAUSS_NODES,GAUSS_WEIGHTS
    q=2*1000.**2/RATE_TAU;cross=sum(w*(expm(A*((s+1)*dt/2))@np.array([0.,0.,1.,0.])) for s,w in zip(nodes,weights))*dt/2*q
    return F,Qj,Qh,cross,q*dt

class SeparatedForceObserver(NormalResponseObserver):
    state_dim=14
    fast_isotropic_control=False
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        blocks(33333,0.);blocks(33333,10.)
    def _initialize(self,z):
        super()._initialize(z);self.P[12:14,12:14]=np.eye(2)*(1000./RATE_TAU)**2

    def _transition(self,dt):
        decay=1/self.correlation_s if self.consistent_dynamics else 0.
        A,Qj,Qh,cross,hvar=blocks(round(dt*1e6),decay)
        F=np.eye(self.state_dim);Q=np.zeros_like(F)
        for axis in range(2):
            ix=[axis,axis+2,axis+4,axis+12]
            F[np.ix_(ix,ix)]=A;Q[np.ix_(ix,ix)]=Qj
        F[:2,self.gain_index]=A[0,2]*self.u;F[2:4,self.gain_index]=A[1,2]*self.u;F[4:6,self.gain_index]=(A[2,2]-1)*self.u
        if self.fast_isotropic_control:
            indices=[0,1,2,3,4,5,12,13];Q[np.ix_(indices,indices)]+=np.kron(Qh,np.eye(2)*.5)
        elif self.normal is not None:
            energy=float(self.normal@self.normal)
            if energy>1e-6:
                Pn=np.outer(self.normal,self.normal)/energy
                indices=[0,1,2,3,4,5,12,13];Q[np.ix_(indices,indices)]+=np.kron(Qh,Pn)
                for axis in range(2):
                    ix=[axis,axis+2,axis+4,axis+12];Q[ix,11]=cross*self.normal[axis]/energy;Q[11,ix]=Q[ix,11]
                Q[11,11]=hvar/energy
        return F,Q

    def _input_covariance_map(self,F):
        B=super()._input_covariance_map(F);B[4:6]=np.eye(2)*(F[4,4]-1.);return B

    def step(self,*args,**kwargs):
        was_two=self.two_point_initialized;c,valid,f,s=super().step(*args,**kwargs)
        if c is not None:
            if self.two_point_initialized and not was_two:self.P[12:14,12:14]=np.eye(2)*(1000./RATE_TAU)**2
            f.update(external_unknown_rate_px_s3=self.x[12:14].tolist(),external_rate_correlation_s=RATE_TAU,
                fast_unknown_force_is_plane_conditioned=True,external_rate_is_measured=False)
        return c,valid,f,s

def rate_kernel(lam,eta,h):
    from adaptive_force_memory import kernel
    if abs(lam-eta)>1e-7:return (kernel(eta,h)-kernel(lam,h))/(eta-lam)
    nodes,weights=GAUSS_NODES,GAUSS_WEIGHTS;s=(nodes+1)*h/2
    return float(np.sum(weights*(h-s)*s*np.exp(lam*s))*h/2)

class IsotropicFastNoiseControl(SeparatedForceObserver):
    """Same trace of fast noise and same mean model, without plane allocation."""
    fast_isotropic_control=True
