"""Visibility-conditioned prior; same rotor physics and fixed-image residual.

This extends the user's4/3/2/line division only in the uncertainty/response
readout, never by inventing hidden rotor measurements. The original mechanical
branch remains. A smooth current/past geometry weight interpolates two response
regimes in one shared model; recording ID, coordinates and future view are not
inputs. This is an engineering hypothesis, not a novelty or accuracy claim.
"""
import copy
import numpy as np
from conditional_unknown_force import ConditionalUnknownForce,teachers,basis,fit
from conic_observability import usable


class VisibilityContext:
    def __init__(self):self.value=None;self.last=None;self.observations=None
    def step(self,row):
        t=row['t'];obs=row.get('state',{}).get('observations')
        reset=bool(self.last is None or t-self.last>.12 or obs is not None and self.observations is not None and obs<self.observations)
        current=[r for r in row['geometry'].get('rotors',[]) if usable(r)] if row['body'].get('detected') else []
        count=min(4,len(current));ratio=float(np.median([r['minor']/r['major'] for r in current])) if current else 0.
        # Existing conic-vs-line degeneracy threshold .16, no new tuned cut.
        # N=2/3/4 gives0/.5/1 completeness context; it is NOT 1/N precision.
        completeness=max(0.,count-2)/2
        raw=completeness*ratio*ratio/(ratio*ratio+.16**2) if ratio else 0.
        if reset:self.value=raw
        else:
            alpha=-np.expm1(-(t-self.last)/.12);self.value+=alpha*(raw-self.value)
        self.last=t;self.observations=obs
        return dict(weight=float(self.value),raw_weight=float(raw),current_qualified_conics=count,
            current_median_axis_ratio=ratio,weight_timescale_s=.12,uses_future_geometry=False,
            count_not_used_as_independent_noise_reduction=True,hidden_rotor_measurements_created=False)


def samples(rows):
    context=VisibilityContext();contexts={r['t']:context.step(r)['weight'] for r in rows}
    return [dict(s,visibility_weight=contexts[s['t']]) for s in teachers(rows,'camera')]


def fit_model(groups):
    nonempty=[g for g in groups if g]
    mean=float(np.mean([np.mean([s['visibility_weight'] for s in g]) for g in nonempty]))
    augmented=[]
    for g in nonempty:
        augmented.append([dict(s,X=np.column_stack((np.asarray(s['X']),(s['visibility_weight']-mean)*np.asarray(s['X']))).tolist()) for s in g])
    model=fit(augmented,'camera',prior=np.zeros(10));model.update(visibility_conditioned=True,
        visibility_context_mean=mean,one_global_model_all_sources=True,current_p_v_a_preserved=True,
        explicit_hidden_rotor_force_owner_unchanged=True,current_geometry_not_heading_used=True)
    return model


class VisibilityResponse:
    def __init__(self,model,constant_context=False):
        self.model=copy.deepcopy(model);self.context=VisibilityContext();self.constant=constant_context
        self.cal=ConditionalUnknownForce(dict(model,weights=model['weights'][:5],visibility_conditioned=False))
    def step(self,row):
        context=self.context.step(row);center=float(self.model['visibility_context_mean']);used=center if self.constant else context['weight']
        theta=np.asarray(self.model['weights']);self.cal.model['weights']=(theta[:5]+(used-center)*theta[5:]).tolist()
        r=self.cal.step(row);r['visibility_response']=dict(context,used_weight=used,
            constant_context_ablation=self.constant,context_center_from_training=center,
            learned_from_future_only_after_maturity_in_training=True,manual_labels_used=False,
            underlying_rotor_prediction_branch_preserved=True,new_body_NN=False)
        return r
