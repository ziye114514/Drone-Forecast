"""Adapt the learned unknown-force continuation from matured native evidence.

No video ID selects an algorithm. Non-overlapping .4s source intervals avoid
counting rolling predictions as independent training examples. Only a current
native, full-FOV rotor envelope can close an old interval; no manual box or
not-yet-observed target is read. Existing source-trained coefficients are a
unit Gaussian prior, updated by robust native displacement likelihoods.
"""
import copy
from collections import deque
import numpy as np
from conditional_unknown_force import ConditionalUnknownForce,basis,fit,base_prediction,HORIZON,TAU
from calibrated_force_rollout import envelope
from separated_force_response import rate_kernel


class MaturedUnknownForce(ConditionalUnknownForce):
    def __init__(self,model):
        super().__init__(copy.deepcopy(model));self.prior=np.array(model['weights']);self.samples=[];self.pending=deque();self.last=None;self.last_t=None
    def step(self,row):
        t=row['t'];current=envelope(row);matured_now=0
        if self.last_t is not None and t-self.last_t>.5:self.pending.clear();self.samples=[];self.model['weights']=self.prior.tolist();self.offset=np.zeros(2);self.last=None
        self.last_t=t
        while self.pending and self.pending[0]['t']+HORIZON<=t+1e-6:
            old=self.pending.popleft();h=t-old['t']
            if current is None or h-HORIZON>.04 or not .6<=current[1]/old['extent']<=1.6:continue
            baseline=np.asarray(base_prediction(old['row'],[h],self.model['mode'])[str(float(h))])-np.asarray(old['row']['coefficient']['position'])
            factor=rate_kernel(-1/TAU,0.,h)/rate_kernel(-1/TAU,0.,HORIZON)
            y=current[0]-old['center']-baseline
            self.samples.append(dict(t=old['t'],matured_at=t,h=h,X=(old['X']*factor).tolist(),y=y.tolist()));matured_now+=1
        if matured_now:
            update=fit([self.samples],self.model['mode'],prior=self.prior,group_normalize=False)
            self.model.update(weights=update['weights'],online_objective=update['objective'])
        if row['valid'] and row['coefficient'] and current is not None and (self.last is None or t-self.last>=HORIZON-1e-6):
            self.pending.append(dict(t=t,row=copy.deepcopy(row),center=current[0],extent=current[1],X=basis(row,self.model['mode']=='camera')));self.last=t
        r=super().step(row)
        if r.get('conditional_unknown_force'):
            r['conditional_unknown_force']['causal_native_adaptation']=dict(matured_samples=len(self.samples),new_samples=matured_now,
                pending_intervals=len(self.pending),min_origin_separation_s=HORIZON,
                newest_target_t=None if not self.samples else self.samples[-1]['matured_at'],
                annotations_used=False,future_target_used_before_maturity=False,offline_weights=self.prior.tolist(),
                evidence_reused_for_response_learning_not_claimed_independent_state_update=True)
        return r
