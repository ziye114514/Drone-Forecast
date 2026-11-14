"""Retain a stochastic past-position clone for direct rotor registration.

The extra clone is STATIC in the process model and retains all cross
covariances. A direct keyframe displacement replaces the adjacent factor,
avoiding artificial velocity impulses from correcting an integrated center.
Force partition, normal response, and both adaptive hypotheses are inherited.
"""
import numpy as np
from adaptive_response_observer import AdaptiveResponseObserver
from separated_force_response import SeparatedForceObserver


class KeyframeClone:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.key_t=None;self.key_used=False;self.key_packet=None

    @property
    def key_indices(self):return slice(self.state_dim-2,self.state_dim)

    def _clone_keyframe(self,t):
        J=np.eye(self.state_dim);J[self.key_indices]=0.
        J[self.key_indices,0:2]=np.eye(2)
        self.x[self.key_indices]=self.x[:2]
        self.P=J@self.P@J.T
        # A persistent template error, NOT 1/N independent corner precision.
        self.P[self.key_indices,self.key_indices]+=np.eye(2)*4.
        self.key_t=float(t)

    def _rebase_observation_reference(self,shift):
        super()._rebase_observation_reference(shift)
        self.key_t=None

    def _measurement(self,z,body,motion,lag):
        H,R,value,flow=super()._measurement(z,body,motion,lag)
        packet=(motion or {}).get('keyframe_rotor_observation') or {}
        if (packet.get('create',True) or not self.relative_used or lag or self.key_t is None or
                abs(packet.get('source_t',-1)-self.key_t)>1e-7):return H,R,value,flow
        delta=np.asarray(packet['displacement_xy'],float);Q=np.asarray(packet['covariance_proxy'],float)
        if delta.shape!=(2,) or Q.shape!=(2,2) or not np.isfinite(delta).all() or not np.isfinite(Q).all():
            raise ValueError('Finite direct-keyframe displacement/covariance required')
        eig,U=np.linalg.eigh((Q+Q.T)/2);Q=(U*np.maximum(eig,4.))@U.T
        H=H.copy();R=R.copy();value=value.copy()
        H[:2]=0.;H[:2,:2]=np.eye(2);H[:2,self.key_indices]=-np.eye(2)
        R[:2,:2]=Q*(2 if self.layout_used else 1);value[:2]=delta
        self.key_used=True
        return H,R,value,flow

    def step(self,t,body,u,*args,**kwargs):
        self.key_used=False
        c,valid,f,s=super().step(t,body,u,*args,**kwargs)
        packet=(kwargs.get('motion') or {}).get('keyframe_rotor_observation') or {}
        if c is not None and packet.get('create') and abs(packet['source_t']-t)<1e-7:
            self._clone_keyframe(t)
        s.update(direct_keyframe_rotor_displacement_used=self.key_used,
            keyframe_clone_t=self.key_t,keyframe_displacement_is_independent_absolute_fix=False)
        return c,valid,f,s


class KeyframeSeparatedObserver(KeyframeClone,SeparatedForceObserver):
    state_dim=16


class KeyframeAdaptiveObserver(KeyframeClone,AdaptiveResponseObserver):
    state_dim=14
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.alternative=KeyframeSeparatedObserver(*args,**kwargs)
