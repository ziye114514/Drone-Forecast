"""Completeness-conditioned two-force state; no duplicate full-thrust scale.

Four CURRENT resolved periods permit the exact coordinate change
 k'=k+h/q, d'=d-h*n, h'=0, u=q*n. Current p/v/net acceleration, external
 residual and their covariance are preserved. The response uncertainty moves
 into k rather than being thrown away. Lost motors still transfer to unknown d.
"""
from collections import deque
import copy
import numpy as np
from normal_response_observer import NormalResponseObserver
from separated_force_response import SeparatedForceObserver,blocks
from causal_response_credit import CausalResponseCredit
from axis_amplitude_bridge import current_axis_qualified


def collapse(x,P,n,u,gain_index=6,response_index=11):
    n=np.asarray(n,float);u=np.asarray(u,float);q=float(n@u/(n@n))
    if q<=0 or not np.allclose(u,q*n,rtol=1e-7,atol=1e-9):return x,P,dict(applied=False)
    k=float(x[gain_index]+x[response_index]/q)
    if k<=0:return x,P,dict(applied=False,reason='positive_force_gain_not_supported')
    J=np.eye(len(x));J[4:6,response_index]=-n;J[gain_index,response_index]=1/q;J[response_index,:]=0.
    return J@x,J@P@J.T,dict(applied=True,relative_current_sum=q,effective_image_gain=k,
        current_state_and_external_force_preserved=True,covariance_transformed_not_reset=True,
        independent_mechanical_calibration=False)


class StructuredForceMixin:
    def __init__(self,*args,**kwargs):
        self.full_anchor=False;self.normal_input_complete=False;self.normal_samples=deque()
        self.incoming_structure={};self.incoming_components={};self.gain_credit=0.;self.last_collapse=dict(applied=False)
        super().__init__(*args,**kwargs)

    def _set_input(self,u,reanchor=False,transfer_delta=None,transfer_cov=None):
        ncur=np.asarray(self.incoming_structure.get('axis_xy') if self.incoming_structure.get('axis_xy') is not None else [],float)
        if transfer_delta is not None and self.normal is not None and ncur.shape==(2,) and ncur@ncur>1e-8:
            if ncur[1]>0:ncur=-ncur
            raw=self.incoming_components
            total=np.sum(list(raw.values()),axis=0) if raw else np.zeros(2)
            new=raw if np.allclose(total,u,atol=1e-9,rtol=1e-9) else {}
            old=self.components;old_keys=set(old);new_keys=set(new);delta=np.zeros(2)
            for key in new_keys-old_keys:
                v=np.asarray(new[key]);w=float(v@ncur/(ncur@ncur))
                # Only true resolved-normal contributions acquire the old-plane
                # transfer basis. An ambiguous pooled vector remains unknown.
                if key.startswith(('slot:','phase_slot:')) and np.allclose(v,w*ncur,rtol=1e-7,atol=1e-9):delta+=w*self.normal
                else:delta+=v
            for key in old_keys-new_keys:delta-=old[key]
            transfer_delta=delta
            if transfer_cov is not None:transfer_cov=np.asarray(transfer_cov)*max(1.,float(self.normal@self.normal/(ncur@ncur)))
        super()._set_input(u,reanchor,transfer_delta,transfer_cov)

    def _gain_learning_allowed(self,good,allow_gain_learning,history_length,excitation,membership_changed,input_meta):
        if not self.full_anchor:return super()._gain_learning_allowed(good,allow_gain_learning,history_length,excitation,membership_changed,input_meta)
        # A full-current-source anchor tied the normal response to the gain.
        # Later plane motion can constrain that model-conditioned coefficient,
        # even after a motor moves to the explicitly unknown contribution.
        # Constant plane/data do not identify it against an arbitrary external
        # force. Only PAST plane samples qualify the current position update.
        if len(self.normal_samples)<8 or self.normal_samples[-1][0]-self.normal_samples[0][0]<.16:
            self.gain_credit=0.;return False
        a=np.asarray([r[1] for r in self.normal_samples]);sigma=np.asarray([r[2] for r in self.normal_samples])
        w=1/np.square(sigma);mean=np.sum(w[:,None]*a,axis=0)/w.sum();energy=float(np.sum(np.square(a-mean)/sigma[:,None]**2))
        dof=2*len(a)-2;self.gain_credit=max(0.,1-dof/max(energy,1e-12))
        return self.gain_credit>0

    def _adjust_gain_update(self,K,residual,dt,excitation):
        if not self.full_anchor:return super()._adjust_gain_update(K,residual,dt,excitation)
        K[self.gain_index]*=-np.expm1(-dt/1.)*self.gain_credit
        if self.x[self.gain_index]+float(K[self.gain_index]@residual)<=0:K[self.gain_index]=0.

    def step(self,t,body,u,*args,**kwargs):
        meta=kwargs.get('input_meta') or {};structure=kwargs.get('structure') or {}
        if self.full_anchor and meta.get('observed_components') and any(not k.startswith(('slot:','phase_slot:')) for k in meta['observed_components']):
            # Physical per-motor calibration does not authorize applying its
            # scale to an unresolved mixture/dominance-change proxy.
            meta=dict(meta,observed_components={},learning_input_current=False,input_sigma_proxy=0.,
                source='unresolved_pool_owned_by_unknown_after_full_anchor')
            kwargs['input_meta']=meta;u=None
        self.incoming_structure=structure;self.incoming_components={k:np.asarray(v,float) for k,v in meta.get('observed_components',{}).items()}
        c,valid,f,s=super().step(t,body,u,*args,**kwargs)
        keys=list(f.get('observed_components',{}));resolved=[k for k in keys if k.startswith(('slot:','phase_slot:'))]
        complete=bool(len(resolved)==4 and len({k.split(':')[-1] for k in resolved})==4 and f.get('active') and self.normal is not None)
        info=dict(applied=False)
        if complete:
            self.x,self.P,info=collapse(self.x,self.P,self.normal,self.u,self.gain_index,self.response_index)
            if info['applied']:
                self.full_anchor=True;self.last_collapse=info
                f.update(effective_gain_px_s2=float(self.x[self.gain_index]),gain_sigma=float(np.sqrt(max(0.,self.P[self.gain_index,self.gain_index]))),
                    disturbance_px_s2=self.x[4:6].tolist(),unobserved_normal_response_gain=0.,unobserved_normal_response_px_s2=[0.,0.],
                    unobserved_normal_response_sigma=0.,external_unknown_not_rotated=self.x[4:6].tolist())
                if f.get('normal_turn_vector_px_s2') is not None:f['normal_turn_vector_px_s2']=(self.x[self.gain_index]*self.u).tolist()
        self.normal_input_complete=bool(complete and self.full_anchor)
        f['structured_rotor_calibration']=dict(full_anchor_seen=self.full_anchor,current_resolved_count=len(resolved),
            current_unknown_normal_collapsed=bool(info['applied']),collapse=info,
            past_plane_gain_credit=float(self.gain_credit),known_gain_is_model_conditioned_not_true_CT=True,
            hidden_current_RPM_measured=0,source_transfers_use_previous_plane_basis=True)
        if current_axis_qualified(structure) and self.normal is not None:
            self.normal_samples.append((t,self.normal.copy(),np.deg2rad(5.)))
        while self.normal_samples and t-self.normal_samples[0][0]>3.:self.normal_samples.popleft()
        return c,valid,f,s


class StructuredNormalForce(StructuredForceMixin,NormalResponseObserver):pass


class StructuredSeparatedForce(StructuredForceMixin,SeparatedForceObserver):
    def _transition(self,dt):
        F,Q=super()._transition(dt)
        if self.normal_input_complete:
            # All four known: external disturbance rate remains, duplicate
            # fast hidden-motor response noise is absent in this interval.
            decay=1/self.correlation_s if self.consistent_dynamics else 0.
            _,Qj,_,_,_=blocks(round(dt*1e6),decay);Q=np.zeros_like(Q)
            for axis in range(2):
                ix=[axis,axis+2,axis+4,axis+12];Q[np.ix_(ix,ix)]=Qj
        return F,Q


class StructuredAdaptiveForce(StructuredNormalForce):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.alternative=StructuredSeparatedForce(*args,**kwargs);self.credit=CausalResponseCredit()

    def step(self,t,body,u,*args,**kwargs):
        c,v,f,s=super().step(t,body,u,*args,**kwargs);self.alternative.two_point_init=self.two_point_init
        ca,va,fa,sa=self.alternative.step(t,body,u,*args,**kwargs)
        common=dict(t=t,body=body,structure=kwargs.get('structure') or {},motion=kwargs.get('motion') or {})
        primary=dict(common,coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'))
        secondary=dict(common,coefficient=ca,valid=va,force=fa,state=sa,normal_turn_forecast=sa.get('normal_turn_forecast'))
        result=self.credit.step(primary,secondary);result['state']['primary_internal_state_is_not_mixture_posterior']=True
        return result['coefficient'],result['valid'],result['force'],result['state']
