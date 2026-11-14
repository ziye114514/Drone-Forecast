"""Fixed-image unknown response conditioned on a current headless rotor plane.

The observed plane supplies a projection, not the direction of body heading.
The added correction remains an uncertain FUTURE net-force-rate prior; visible
measured contributions and all hidden-force ownership stay in the old ledger.
"""
import copy
import numpy as np
from axis_amplitude_bridge import current_axis_qualified
from adaptive_force_memory import kernel
from separated_force_response import rate_kernel
from conditional_unknown_force import fit,HORIZON,TAU
from visibility_response import samples,VisibilityResponse


def plane_basis(row):
    n=np.asarray(row['structure'].get('axis_xy') or [],float)
    if not row.get('coefficient') or n.shape!=(2,) or np.linalg.norm(n)<1e-6 or not current_axis_qualified(row['structure']):return np.zeros((2,2))
    P=np.outer(n,n)/float(n@n);c=row['coefficient']
    return np.column_stack((HORIZON*P@np.asarray(c['velocity']),kernel(-1/TAU,HORIZON)*P@np.asarray(c['acceleration'])))


def plane_samples(rows):
    lookup={r['t']:r for r in rows};out=[]
    for s in samples(rows):
        factor=rate_kernel(-1/TAU,0.,s['h'])/rate_kernel(-1/TAU,0.,HORIZON)
        out.append(dict(s,plane_X=(plane_basis(lookup[s['t']])*factor).tolist()))
    return out


def fit_model(groups,collective_drive=False):
    nonempty=[g for g in groups if g];mean=float(np.mean([np.mean([s['visibility_weight'] for s in g]) for g in nonempty]));augmented=[]
    for g in nonempty:
        augmented.append([dict(s,X=np.column_stack((np.asarray(s['X']),(s['visibility_weight']-mean)*np.asarray(s['X']),s['plane_X'])).tolist()) for s in g])
    if collective_drive:
        for g in augmented:
            for s in g:
                s['X']=np.column_stack((s['X'],s['drive_feature'])).tolist();s['y']=(np.asarray(s['y'])+s['drive_feature']).tolist()
    model=fit(augmented,'camera',prior=np.r_[np.zeros(12),1.] if collective_drive else np.zeros(12),
        bounds=[(None,None)]*12+[(0.,None)] if collective_drive else None)
    if collective_drive:
        model['future_drive_multiplier']=model['weights'].pop();model['prior_weights'].pop()
    model.update(visibility_conditioned=True,visibility_context_mean=mean,
        plane_conditioned=True,one_global_model_all_sources=True,current_p_v_a_preserved=True,
        explicit_hidden_rotor_force_owner_unchanged=True,headless_normal_projection=True)
    return model


class PlaneResponse:
    def __init__(self,model,disabled=False,fixed_gamma=None):
        self.model=copy.deepcopy(model);self.visibility=VisibilityResponse(dict(model,weights=model['weights'][:10]));self.disabled=disabled;self.fixed_gamma=fixed_gamma
    def step(self,row):
        if 'future_drive_multiplier' in self.model:
            row=copy.deepcopy(row);row['future_drive_response']=dict(multiplier=float(self.model['future_drive_multiplier'] if self.fixed_gamma is None else self.fixed_gamma),
                present_acceleration_and_force_ledger_unchanged=True,metric_thrust_measured=False,constant_gamma_ablation=self.fixed_gamma is not None)
        r=self.visibility.step(row)
        if r.get('coefficient') is None:return r
        extra=plane_basis(row)@np.asarray(self.model['weights'][10:]) if not self.disabled else np.zeros(2)
        info=r['conditional_unknown_force'];j=np.asarray(info['future_unknown_rate_px_s3'])+extra/rate_kernel(-1/TAU,0.,HORIZON)
        info.update(future_unknown_rate_px_s3=j.tolist(),plane_conditioned=True)
        r['plane_conditioned_response']=dict(disabled=self.disabled,headless_projection_used=True,
            projected_extra_displacement_at04_px=extra.tolist(),extra_is_not_measured_thrust=True,
            current_p_v_a_and_force_ledger_unchanged=True)
        from physical_publication import evaluate
        r['physical']=evaluate(r,[.1,.2,.4]);return r
