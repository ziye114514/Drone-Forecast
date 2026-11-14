"""Causal motion of the visible-envelope REFERENCE, separate from body force.

The native track's material reference and the scored visible rotor enclosure
are different points. Current/past enclosure offsets may move with attitude or
visibility. Only this reference motion is extrapolated; present state and the
visible/hidden force ledger remain unchanged. No future geometry is supplied.
"""
import copy
import numpy as np
from calibrated_force_rollout import envelope
from visibility_response import samples,VisibilityResponse
from conditional_unknown_force import fit


class ReferenceMotion:
    def __init__(self):self.rows=[];self.observations=None
    def step(self,row):
        t=float(row['t']);obs=row.get('state',{}).get('observations')
        if obs is not None and self.observations is not None and obs<self.observations:self.rows=[]
        self.observations=obs
        if self.rows and t-self.rows[-1][0]>.12:self.rows=[]
        current=envelope(row) if row.get('coefficient') else None
        if current is not None:
            self.rows.append((t,current[0]-np.asarray(row['coefficient']['position'])))
        self.rows=[q for q in self.rows if t-q[0]<=.35]
        if current is None or len(self.rows)<4 or self.rows[-1][0]-self.rows[0][0]<.12:
            return dict(qualified=False,velocity_xy=[0.,0.],rows=len(self.rows))
        times=np.array([q[0]-t for q in self.rows]);y=np.array([q[1] for q in self.rows]);w=np.exp(times/.35)
        mean=np.average(times,weights=w);centered=times-mean
        velocity=np.sum(w[:,None]*centered[:,None]*y,axis=0)/float(np.sum(w*centered**2))
        return dict(qualified=True,velocity_xy=velocity.tolist(),rows=len(self.rows),span_s=float(times[-1]-times[0]),
            material_reference_is_not_visible_envelope=True,future_geometry_used=False)


def reference_samples(rows):
    motion=ReferenceMotion();contexts={r['t']:motion.step(r) for r in rows}
    return [dict(s,reference_velocity=contexts[s['t']]['velocity_xy']) for s in samples(rows)]


def fit_model(groups):
    nonempty=[g for g in groups if g];mean=float(np.mean([np.mean([s['visibility_weight'] for s in g]) for g in nonempty]));augmented=[]
    for g in nonempty:
        augmented.append([dict(s,X=np.column_stack((np.asarray(s['X']),(s['visibility_weight']-mean)*np.asarray(s['X']),s['h']*np.asarray(s['reference_velocity']))).tolist()) for s in g])
    model=fit(augmented,'camera',prior=np.zeros(11),bounds=[(None,None)]*10+[(0.,1.)])
    model['reference_velocity_weight']=model['weights'].pop();model['prior_weights'].pop()
    model.update(visibility_conditioned=True,visibility_context_mean=mean,reference_motion_conditioned=True,
        one_global_model_all_sources=True,current_p_v_a_preserved=True,reference_motion_is_not_force=True,
        explicit_hidden_rotor_force_owner_unchanged=True)
    return model


class ReferenceResponse:
    def __init__(self,model,disabled=False):
        self.model=copy.deepcopy(model);self.motion=ReferenceMotion();self.visibility=VisibilityResponse(model);self.disabled=disabled
    def step(self,row):
        motion=self.motion.step(row);r=self.visibility.step(row)
        r['native_reference_motion']=dict(motion,weight=0. if self.disabled else float(self.model['reference_velocity_weight']),
            disabled_reference_ablation=self.disabled,current_physical_state_and_force_unchanged=True)
        from physical_publication import evaluate
        r['physical']=evaluate(r,[.1,.2,.4]);return r


def predict(row,horizons):
    from physical_publication import evaluate
    r=copy.deepcopy(row);info=r.pop('native_reference_motion');values=evaluate(r,horizons)
    velocity=float(info['weight'])*np.asarray(info['velocity_xy'])
    return {str(float(h)):(np.asarray(values[str(float(h))])+float(h)*velocity).tolist() for h in horizons}
