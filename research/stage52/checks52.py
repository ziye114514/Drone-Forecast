import numpy as np
from feedback52 import *

def inputs(t):
    n=np.array([.35*np.sin(t*2),-.8,.45]);n/=np.linalg.norm(n)
    return np.array([10*t*t,-8*t*t,20*t,-16*t]),n,1+.15*np.sin(t*3)
def run():
    cfg=prior.PaperConfig();old=prior.ProjectedThrustModel(cfg,True);new=DelayedFeedback(cfg,True)
    for j in range(100):
        t=j*.02;state,n,s=inputs(t)
        old.update(t,state,n,s,.9,True,True);new.update(t,state,n,s,.9,True,True)
        assert old.updates==new.updates
        np.testing.assert_array_equal(old.parameters,new.parameters);np.testing.assert_array_equal(old.cov,new.cov);np.testing.assert_array_equal(old.disturbance,new.disturbance)
    a=prior.ProjectedThrustModel(cfg,True);b=DelayedFeedback(cfg,True);missing_position=DelayedFeedback(cfg,True)
    before=None;launched=0
    for j in range(75):
        t=j*.02;state,n,s=inputs(t);visible=t<.82
        for model in [a,b,missing_position]:
            model.update(t,state,n if visible else None,s if visible else 1.,.9 if visible else 0.,visible or model is not missing_position,visible)
        if visible:before=(a.updates,b.updates,missing_position.updates);launched=len(b.queue)
        else:assert not b.update_allowed
    assert b.updates>a.updates and a.updates==before[0] and missing_position.updates==before[2]
    assert not b.queue and not missing_position.queue
    after=b.updates
    for j in range(75,100):
        t=j*.02;state,n,s=inputs(t);b.update(t,state,None,1.,0.,True,False)
    assert b.updates==after
    return dict(fully_observed_parameters_covariance_residual_exact=True,fully_observed_updates=new.updates,
        occlusion_original_updates=a.updates,occlusion_delayed_updates=b.updates,no_current_position_updates=missing_position.updates,
        no_new_features_from_missing_rotors=True,queued_samples_expire=True,no_repeated_updates_after_queue_empty=True,
        limits='Controlled state feedback, not measured physical dynamics accuracy.')
