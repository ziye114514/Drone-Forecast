"""Two force-response hypotheses, shared rotor observations and causal credit.

Primary191 and separated201 do not provide independent observations. The
inherited primary state remains the reference/entry bookkeeping state; output
moments and forecasts are explicitly a mixture, never an independent fusion.
"""
from normal_response_observer import NormalResponseObserver
from separated_force_response import SeparatedForceObserver
from causal_response_credit import CausalResponseCredit

class AdaptiveResponseObserver(NormalResponseObserver):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.alternative=SeparatedForceObserver(*args,**kwargs);self.credit=CausalResponseCredit()

    def step(self,t,body,u,*args,**kwargs):
        c,valid,f,s=super().step(t,body,u,*args,**kwargs)
        self.alternative.two_point_init=self.two_point_init
        ca,va,fa,sa=self.alternative.step(t,body,u,*args,**kwargs)
        common=dict(t=t,body=body,structure=kwargs.get('structure') or {},motion=kwargs.get('motion') or {})
        base=dict(common,coefficient=c,valid=valid,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'))
        alt=dict(common,coefficient=ca,valid=va,force=fa,state=sa,normal_turn_forecast=sa.get('normal_turn_forecast'))
        result=self.credit.step(base,alt)
        result['state']['primary_internal_state_is_not_mixture_posterior']=True
        return result['coefficient'],result['valid'],result['force'],result['state']
