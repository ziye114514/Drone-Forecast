"""Allocate the SAME net-disturbance noise to visible-plane force directions.

191 keeps h constant and puts every process increment in free2D d. Here the
normal projection of that SAME increment is shared with h, while e=d-h*n
gets the perpendicular part. Q[p,v,d] is unchanged; no extra noise is added.
Absolute hidden thrust/gravity are still not uniquely identifiable.
"""
import numpy as np
from normal_response_observer import NormalResponseObserver
from normal_damping_observer import kernels

NODES,WEIGHTS=np.polynomial.legendre.leggauss(8)

class DynamicNormalResponseObserver(NormalResponseObserver):
    def _transition(self,dt):
        F,Q=super()._transition(dt)
        if self.normal is None:return F,Q
        energy=float(self.normal@self.normal)
        if energy<=1e-6:return F,Q
        elapsed=round(dt*1e6)*1e-6 if self.consistent_dynamics else dt
        rate=1/self.correlation_s if self.consistent_dynamics else 0.
        s=(NODES+1)*elapsed/2;weights=WEIGHTS*elapsed/2
        response=np.array([[kernels(rate,t)[1],kernels(rate,t)[0],np.exp(-rate*t)] for t in s])
        spectral=2*1000.**2/self.correlation_s
        cross=(weights@response)*spectral
        h=self.response_index
        Q[:6,h]=np.kron(cross,self.normal/energy);Q[h,:6]=Q[:6,h]
        Q[h,h]=spectral*elapsed/energy
        return F,Q

    def step(self,*args,**kwargs):
        c,valid,f,s=super().step(*args,**kwargs)
        f.update(unknown_normal_amplitude_process='same_net_noise_projected_on_current_plane',
            total_net_disturbance_noise_increased=False,absolute_hidden_thrust_identified=False)
        return c,valid,f,s
