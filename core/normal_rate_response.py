"""Unknown normal response has a rate, not just white amplitude variation.

Same14states and same noise intensities as separated201. Project the existing
unknown acceleration rate onto the CURRENT plane. Its accumulated normal part
updates h; the perpendicular part remains external. This is a structural prior,
not separate measurement of gravity/wind/thrust or mechanical RPM.
"""
import copy
import numpy as np
from separated_force_response import SeparatedForceObserver,blocks,RATE_TAU,rate_kernel
from normal_response_observer import NormalResponseObserver
from causal_response_credit import CausalResponseCredit
from physical_publication import evaluate as original_evaluate
from adaptive_force_memory import kernel
from axis_amplitude_bridge import current_axis_qualified


class NormalRateSeparated(SeparatedForceObserver):
    def _transition(self,dt):
        F,Q=super()._transition(dt)
        if self.normal is None:return F,Q
        energy=float(self.normal@self.normal)
        if energy<1e-6:return F,Q
        decay=1/self.correlation_s if self.consistent_dynamics else 0.
        A,Qj,_,_,_=blocks(round(dt*1e6),decay)
        direction=self.normal/energy
        F[11,12:14]=A[2,3]*direction
        # h receives the SAME integrated normal component of the existing j
        # noise. No new diffusion, state or fitted variance is introduced.
        for axis in range(2):
            indices=[axis,axis+2,axis+4,axis+12]
            cross=Qj[:,2]*direction[axis]
            Q[indices,11]+=cross;Q[11,indices]+=cross
        Q[11,11]+=Qj[2,2]/energy
        return F,Q

    def step(self,t,body,u,*args,**kwargs):
        c,valid,f,s=super().step(t,body,u,*args,**kwargs)
        structure=kwargs.get('structure') or {}
        qualified=self.normal is not None and current_axis_qualified(structure)
        if c is not None and qualified:
            n=self.normal;energy=float(n@n)
            normal_rate=n*float(n@self.x[12:14])/energy
            f.update(unknown_normal_response_rate_px_s3=normal_rate.tolist(),
                current_total_unknown_rate_px_s3=self.x[12:14].tolist(),
                rate_partition_is_model_conditioned=True,
                normal_response_rate_is_measured_thrust_rate=False,
                external_rate_field_includes_normal_part_before_rollout_correction=True)
        s.update(unknown_normal_response_has_integrated_rate=True,
            additional_state_count=0,noise_intensities_changed=False)
        return c,valid,f,s


NODES,WEIGHTS=np.polynomial.legendre.leggauss(16)


def normal_rate_kernel(lam,eta,omega,h):
    if not omega:return complex(rate_kernel(lam,eta,h))
    if abs(lam-eta)>1e-7:return (kernel(eta+1j*omega,h)-kernel(lam+1j*omega,h))/(eta-lam)
    s=(NODES+1)*h/2
    return complex(np.sum(WEIGHTS*(h-s)*s*np.exp((lam+1j*omega)*s))*h/2)


def evaluate(row,horizons):
    horizons=list(horizons);mixture=row['force'].get('response_model_mixture')
    if mixture:
        values={str(float(h)):np.zeros(2) for h in horizons}
        for component in mixture:
            part=evaluate(component,horizons)
            for h,p in part.items():values[h]+=component['weight']*np.asarray(p)
        return {h:p.tolist() for h,p in values.items()}
    result=original_evaluate(row,horizons);f=row['force'];jn=f.get('unknown_normal_response_rate_px_s3')
    turn=row.get('normal_turn_forecast') or {}
    if jn is None or not turn.get('applied'):return result
    if (turn.get('projected_tilt_trend') or {}).get('qualified') or turn.get('trend',{}).get('angular_acceleration_rad_s2'):
        raise ValueError('This frozen rate pilot does not combine with other tilt/curvature candidates')
    omega=turn['trend']['shrunk_angular_rate_rad_s'];lam=-1/float(f['acceleration_correlation_s']);eta=-1/float(f['external_rate_correlation_s'])
    z=complex(*jn)
    for h,p in result.items():
        delta=(normal_rate_kernel(lam,eta,omega,float(h))-normal_rate_kernel(lam,eta,0.,float(h)))*z
        result[h]=(np.asarray(p)+[delta.real,delta.imag]).tolist()
    return result


class RateCredit(CausalResponseCredit):
    def step(self,base,alt):
        # The credit module imports its rollout at module scope. Keep a private
        # copy of its function globals, rather than changing other predictors.
        import types
        method=CausalResponseCredit.step
        private=dict(method.__globals__,evaluate=evaluate)
        call=types.FunctionType(method.__code__,private,argdefs=method.__defaults__)
        return call(self,base,alt)


class NormalRateAdaptive(NormalResponseObserver):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.alternative=NormalRateSeparated(*args,**kwargs);self.credit=RateCredit()
    def step(self,t,body,u,*args,**kwargs):
        c,valid,f,s=super().step(t,body,u,*args,**kwargs)
        self.alternative.two_point_init=self.two_point_init
        ca,va,fa,sa=self.alternative.step(t,body,u,*args,**kwargs)
        common=dict(t=t,body=body,structure=kwargs.get('structure') or {},motion=kwargs.get('motion') or {})
        base=dict(common,coefficient=c,valid=valid,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'))
        alt=dict(common,coefficient=ca,valid=va,force=fa,state=sa,normal_turn_forecast=sa.get('normal_turn_forecast'))
        r=self.credit.step(base,alt);r['state']['primary_internal_state_is_not_mixture_posterior']=True
        return r['coefficient'],r['valid'],r['force'],r['state']
