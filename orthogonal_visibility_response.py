"""Separate amount of visible conic support from plane degeneracy.

Two well-resolved ellipses are not the same information state as a thin side
line. This separates those roles, without claiming extra hubs or reducing
noise by N. Both contexts remain current/past-only, with the same.16/.12
constants. One15-scalar residual model, no recording-specific rules or NN.
"""
import copy
import numpy as np
from visibility_response import VisibilityContext
from conditional_unknown_force import ConditionalUnknownForce,teachers,fit


class SeparateVisibilityContext:
    def __init__(self):self.base=VisibilityContext();self.value=None;self.last=None;self.obs=None
    def step(self,row):
        old=self.base.step(row);n=old['current_qualified_conics'];ratio=old['current_median_axis_ratio'];t=row['t'];obs=row.get('state',{}).get('observations')
        raw=np.array([ratio*ratio/(ratio*ratio+.16**2) if n>=2 else 0.,n/4.])
        reset=self.last is None or t-self.last>.12 or obs is not None and self.obs is not None and obs<self.obs
        if reset:self.value=raw
        else:self.value+=-np.expm1(-(t-self.last)/.12)*(raw-self.value)
        self.last=t;self.obs=obs
        return dict(plane_geometry_weight=float(self.value[0]),visible_support_weight=float(self.value[1]),
            raw_plane_geometry_weight=float(raw[0]),raw_visible_support_weight=float(raw[1]),
            current_qualified_conics=n,median_axis_ratio=ratio,
            fewer_conics_not_automatically_edge_on=True,hidden_rotor_or_independent_noise_information_created=False)


def samples(rows):
    m=SeparateVisibilityContext();context={}
    for r in rows:
        c=m.step(r);context[r['t']]=[c['plane_geometry_weight'],c['visible_support_weight']]
    return [dict(s,context=context[s['t']]) for s in teachers(rows,'camera')]


def fit_model(groups):
    groups=[g for g in groups if g];mean=np.mean([np.mean([s['context'] for s in g],axis=0) for g in groups],axis=0);aug=[]
    for g in groups:
        aug.append([dict(s,X=np.column_stack([np.asarray(s['X'])]+[(v-m)*np.asarray(s['X']) for v,m in zip(s['context'],mean)]).tolist()) for s in g])
    model=fit(aug,'camera',prior=np.zeros(15));model.update(separate_visibility_conditioned=True,
        visibility_context_mean=mean.tolist(),one_shared_model=True,no_hyperparameter_scan=True,
        geometric_degeneracy_and_visible_count_not_conflated=True)
    return model


class SeparateVisibilityResponse:
    def __init__(self,model,ablation='none'):
        self.model=copy.deepcopy(model);self.context=SeparateVisibilityContext();self.ablation=ablation
        self.cal=ConditionalUnknownForce(dict(model,weights=model['weights'][:5],separate_visibility_conditioned=False))
    def step(self,row):
        context=self.context.step(row);used=np.array([context['plane_geometry_weight'],context['visible_support_weight']]);center=np.asarray(self.model['visibility_context_mean'])
        if self.ablation in ('no_geometry','constant'):used[0]=center[0]
        if self.ablation in ('no_count','constant'):used[1]=center[1]
        theta=np.asarray(self.model['weights']);effective=theta[:5]+(used[0]-center[0])*theta[5:10]+(used[1]-center[1])*theta[10:15]
        self.cal.model['weights']=effective.tolist();r=self.cal.step(row)
        r['separate_visibility_response']=dict(context,used_context=used.tolist(),ablation=self.ablation,
            underlying_rotor_force_branch_preserved=True,frame='fixed image',current_p_v_a_unchanged=True)
        return r
