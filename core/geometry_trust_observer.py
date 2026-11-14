"""Causal trust in the plane constraint on the unobservable force.

Both hypotheses use the SAME rotor-only observations and measured term k*u.
One leaves d free; one uses the existing plane-adaptive d=e+h*n. They are
correlated model alternatives, never independent sensors or NN trajectories.
"""
from collections import deque
import copy
import numpy as np
from adaptive_force_memory import AdaptiveForceMemory
from adaptive_response_observer import AdaptiveResponseObserver
from layout_reference_observer import LayoutReferenceObserver
from causal_response_credit import CausalResponseCredit


class RetainPlanePriorMemory(AdaptiveForceMemory):
    """Same matured-error robust fit, ridge prior at the existing plane model."""
    def __init__(self,extend_calibrated=False,allow_shorter=False):
        super().__init__(extend_calibrated,allow_shorter);self.weight=1.

    def step(self,*args,**kwargs):
        previous_weight=self.weight
        result,info=super().step(*args,**kwargs)
        if info['updated']:
            t=args[0] if args else kwargs['t'];numerator=36.;denominator=36.
            for stamp,delta,error,h in self.rows:
                w=np.exp(-(t-stamp)/.35)/3.
                residual=error-previous_weight*delta;robust=min(1.,6./max(float(np.linalg.norm(residual)),1e-9))
                numerator+=w*robust*float(delta@error);denominator+=w*robust*float(delta@delta)
            self.weight=float(np.clip(numerator/denominator,0.,1.))
        return result,dict(info,weight=self.weight,prior='retain existing plane constraint until matured evidence contradicts it')


class GeometryTrustObserver:
    requires_current_structure=True
    def __init__(self,*args,**kwargs):
        self.plane=AdaptiveResponseObserver(*args,**kwargs);self.free=LayoutReferenceObserver(*args,**kwargs)
        self.credit=CausalResponseCredit();self.credit.memory=RetainPlanePriorMemory();self.two_point_init=False

    def step(self,t,body,u,*args,**kwargs):
        self.plane.two_point_init=self.free.two_point_init=self.two_point_init
        cp,vp,fp,sp=self.plane.step(t,body,u,*args,**kwargs)
        free_kwargs=dict(kwargs);structure=free_kwargs.pop('structure',{}) or {}
        cf,vf,ff,sf=self.free.step(t,body,u,*args,**free_kwargs)
        common=dict(t=t,body=body,motion=kwargs.get('motion') or {},structure=structure)
        base=dict(common,coefficient=cf,valid=vf,force=ff,state=sf,normal_turn_forecast=sf.get('normal_turn_forecast'))
        plane=dict(common,coefficient=cp,valid=vp,force=fp,state=sp,normal_turn_forecast=sp.get('normal_turn_forecast'))
        result=self.credit.step(base,plane);info=result['state'].pop('causal_response_credit')
        info.update(starts_with_original191=False,prior_plane_weight=1.,
            free_branch_is_same_observed_plus_unknown_force=True,NN_branch=False,
            selector_uses_recording_identity=False,independent_tracks_fused=False)
        result['state']['causal_geometry_trust']=info
        result['state']['plane_response_credit']=sp.get('causal_response_credit')
        return result['coefficient'],result['valid'],result['force'],result['state']
