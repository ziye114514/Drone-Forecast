"""One nonlinear unknown-rate prior, with explicit projection assumptions.

Quadratic drag is proportional to -|v_air|v_air in metric space. Under locally
fixed projection scale and a still-air image-velocity proxy, differentiate
-|v|v/extent. Its CURRENT value is already inside the net acceleration; only
its future change is added, avoiding double-counting present force. Neither
wind, metric drag, hidden RPM nor mass is identified by this approximation.
"""
import copy
import numpy as np
from calibrated_force_rollout import envelope
from separated_force_response import rate_kernel
from conditional_unknown_force import fit,HORIZON,TAU
from visibility_response import VisibilityResponse


def drag_rate(row):
    current=envelope(row) if row.get('coefficient') else None
    if current is None:return np.zeros(2)
    c=row['coefficient'];v=np.asarray(c['velocity']);a=np.asarray(c['acceleration']);speed=float(np.linalg.norm(v))
    if speed<1e-9:return np.zeros(2)
    return -(speed*a+v*float(v@a)/speed)/current[1]


def fit_model(groups,enabled=True):
    nonempty=[g for g in groups if g];mean=float(np.mean([np.mean([s['visibility_weight'] for s in g]) for g in nonempty]));augmented=[]
    for g in nonempty:
        gg=[]
        for s in g:
            X=np.column_stack((s['X'],(s['visibility_weight']-mean)*np.asarray(s['X'])))
            if enabled:X=np.column_stack((X,s['drag_feature']))
            gg.append(dict(s,X=X.tolist()))
        augmented.append(gg)
    model=fit(augmented,'camera',prior=np.zeros(11 if enabled else 10),bounds=[(None,None)]*10+[(0.,None)] if enabled else None)
    model['quadratic_unknown_rate_weight']=model['weights'].pop() if enabled else 0.
    if enabled:model['prior_weights'].pop()
    model.update(visibility_conditioned=True,visibility_context_mean=mean,quadratic_unknown_rate_conditioned=enabled,
        image_projection_and_still_air_assumptions=True,physical_drag_measured=False,
        one_global_model_all_sources=True,current_p_v_a_preserved=True,explicit_hidden_rotor_force_owner_unchanged=True)
    return model


class QuadraticResponse:
    def __init__(self,model,disabled=False):self.model=copy.deepcopy(model);self.visibility=VisibilityResponse(model);self.disabled=disabled
    def step(self,row):
        r=self.visibility.step(row)
        if r.get('coefficient') is None:return r
        extra=drag_rate(row)*float(self.model['quadratic_unknown_rate_weight']) if not self.disabled else np.zeros(2)
        info=r['conditional_unknown_force'];info['future_unknown_rate_px_s3']=(np.asarray(info['future_unknown_rate_px_s3'])+extra).tolist()
        r['quadratic_unknown_response']=dict(disabled=self.disabled,added_unknown_rate_px_s3=extra.tolist(),
            current_force_not_counted_twice=True,metric_drag_measured=False,still_air_projection_proxy=True)
        from physical_publication import evaluate
        r['physical']=evaluate(r,[.1,.2,.4]);return r
