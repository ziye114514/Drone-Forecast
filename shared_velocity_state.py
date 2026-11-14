"""Commit the already fused current velocity into the one physical state.

The image velocity is correlated with localization. A joint covariance bound
2*diag(P_position, P_velocity_CI) covers any admissible unknown cross block;
we do not pretend the two marginal estimates are independent. There is no
extra filter, acceleration command, detector, or forecast branch here.
"""
import numpy as np


def commit_velocity(kf,coefficient,motion):
    info=dict(applied=False,reason='no_current_fused_velocity',new_filters=0,
        acceleration_not_fed_as_control=True,covariance_is_engineering_proxy=True)
    velocity=(motion or {}).get('velocity') or {}
    observation=(motion or {}).get('observation') or {}
    if kf.x is None or coefficient is None or not velocity.get('applied') or not observation.get('valid'):
        return info
    w=float(velocity['baseline_weight']);P=np.asarray(kf.P)
    R=np.asarray(observation['velocity_covariance'])
    C=np.linalg.inv(w*np.linalg.inv(P[2:4,2:4])+(1-w)*np.linalg.inv(R))
    before=kf.x[2:4].copy();kf.x[2:4]=np.asarray(coefficient['velocity'])
    bounded=np.zeros((4,4));bounded[:2,:2]=2*P[:2,:2];bounded[2:4,2:4]=2*C
    kf.P=(bounded+bounded.T)*.5
    return dict(info,applied=True,reason='commit_existing_CI_velocity_with_unknown_cross_covariance_bound',
        velocity_before=before.tolist(),velocity_after=kf.x[2:4].tolist(),
        covariance_bound='2 * blockdiag(position marginal, CI velocity marginal)',
        position_mean_unchanged=True)
