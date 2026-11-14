import numpy as np
from reference51 import *

def run():
    cfg=prior.PaperConfig();a=prior.ProjectedThrustModel(cfg,True);b=prior.ProjectedThrustModel(cfg,True);ref=Reference();maximum=0.
    for j in range(100):
        t=j*.02;S=400000.*(1+.15*np.sin(t*3));denom=400000. if j<35 else (700000. if j<70 else 250000.)
        sensed=dict(four_frequencies=True,thrust_ratio=S/denom,sum_frequency_squared=S)
        mapped,d=ref.transform(sensed);s=S/400000.;np.testing.assert_allclose(mapped['thrust_ratio'],s,rtol=1e-14)
        n=np.array([.35*np.sin(t*2),-.8,.45]);n/=np.linalg.norm(n);state=np.array([10*t*t,-8*t*t,20*t,-16*t])
        a.update(t,state,n,s,.9,True,True);b.update(t,state,n,mapped['thrust_ratio'],.9,True,True)
        np.testing.assert_allclose(a.parameters,b.parameters,rtol=1e-10,atol=1e-10);np.testing.assert_allclose(a.cov,b.cov,rtol=1e-10,atol=1e-10)
        maximum=max(maximum,float(np.max(abs(a.parameters-b.parameters))))
        assert len(a.queue)==len(b.queue) and a.updates==b.updates
        for qa,qb in zip(a.queue,b.queue):np.testing.assert_allclose(qa[4],qb[4],rtol=1e-14)
        ta,pa=a.rollout(state[:2],state[2:],n,np.zeros(3),s);tb,pb=b.rollout(state[:2],state[2:],n,np.zeros(3),mapped['thrust_ratio'])
        np.testing.assert_allclose(pa,pb,rtol=1e-10,atol=1e-10)
    assert a.updates>0
    # Explicit parameter/covariance/old-regressor transformation equivalence.
    f=1.75;D=np.diag([f,1.,1.,1.]);n=np.array([.2,-.8]);v=np.array([12.,-15.]);s=1.2;up=np.asarray(cfg.image_up)
    X=np.array([[s*n[0],-up[0],-v[0],0.],[s*n[1],-up[1],0.,-v[1]]]);Y=X@np.linalg.inv(D)
    np.testing.assert_allclose(Y@(D@a.parameters),X@a.parameters,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(Y@(D@a.cov@D.T)@Y.T,X@a.cov@X.T,rtol=1e-12,atol=1e-12)
    ref.clear();assert ref.total is None
    ref.total=1.;q,d=ref.transform(dict(four_frequencies=True,thrust_ratio=1.,sum_frequency_squared=20.));assert d['range_rejected'] and not q['four_frequencies']
    return dict(canonical_rebased_rls_updates=a.updates,maximum_parameter_difference=maximum,
        learned_parameters_covariance_queue_and_rollouts_invariant=True,explicit_regressor_parameter_covariance_invariant=True,
        reference_clear_and_out_of_domain_abstention=True,original_rls_bounds_excitation_and_forgetting_executed=True)
