"""One training-only collective future-drive calibration, preserving present state.

This does not identify thrust in Newtons. Gamma scales only the FUTURE effect
of a rotating collective normal drive, with the present net acceleration held
fixed. Hidden rotor contributions remain in the lumped unknown. No heading,
motor identity, recording ID or future observations are runtime inputs.
"""
import copy
import numpy as np
from visibility_response import samples,VisibilityResponse
from conditional_unknown_force import fit


def scaled_packet(row,gamma):
    r=copy.deepcopy(row);r.pop('future_drive_response',None)
    def change(p):
        f=p['force']
        if f.get('normal_turn_vector_px_s2') is not None:
            f['normal_turn_vector_px_s2']=(gamma*np.asarray(f['normal_turn_vector_px_s2'])).tolist()
        if f.get('anisotropic_disk_response'):
            f['anisotropic_disk_response']['normal_drive_px_s2']*=gamma
        for part in f.get('response_model_mixture',[]):change(part)
    change(r);return r


def drive_feature(row,h):
    from physical_publication import evaluate
    return np.asarray(evaluate(row,[h])[str(float(h))])-np.asarray(evaluate(scaled_packet(row,0.),[h])[str(float(h))])


def drive_samples(rows):
    lookup={r['t']:r for r in rows};out=[]
    for s in samples(rows):
        d=drive_feature(lookup[s['t']],s['h'])
        out.append(dict(s,drive_feature=d.tolist()))
    return out


def fit_model(groups):
    nonempty=[g for g in groups if g]
    mean=float(np.mean([np.mean([s['visibility_weight'] for s in g]) for g in nonempty]));augmented=[]
    for g in nonempty:
        augmented.append([dict(s,X=np.column_stack((np.asarray(s['X']),(s['visibility_weight']-mean)*np.asarray(s['X']),s['drive_feature'])).tolist(),
            y=(np.asarray(s['y'])+s['drive_feature']).tolist()) for s in g])
    solved=fit(augmented,'camera',prior=np.r_[np.zeros(10),1.],bounds=[(None,None)]*10+[(0.,None)])
    gamma=solved['weights'].pop();solved['prior_weights'].pop()
    solved.update(visibility_conditioned=True,visibility_context_mean=mean,
        future_drive_multiplier=gamma,drive_prior_mean=1.,drive_prior_std=1.,drive_nonnegative=True,
        one_global_model_all_sources=True,current_p_v_a_preserved=True,
        explicit_hidden_rotor_force_owner_unchanged=True,current_geometry_not_heading_used=True)
    return solved


class FutureDriveResponse:
    def __init__(self,model,fixed_gamma=None):
        self.model=copy.deepcopy(model);self.visibility=VisibilityResponse(model);self.fixed=fixed_gamma
    def step(self,row):
        r=copy.deepcopy(row)
        r['future_drive_response']=dict(multiplier=float(self.model['future_drive_multiplier'] if self.fixed is None else self.fixed),
            present_acceleration_and_force_ledger_unchanged=True,metric_thrust_measured=False,
            constant_gamma_ablation=self.fixed is not None)
        return self.visibility.step(r)


def predict(row,horizons):
    from physical_publication import evaluate
    return evaluate(scaled_packet(row,float(row['future_drive_response']['multiplier'])),horizons)
