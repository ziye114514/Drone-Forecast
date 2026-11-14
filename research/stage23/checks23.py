import numpy as np
from robust23 import cap,robust_acceleration,Predictor,PARAMS


def run():
    np.testing.assert_allclose(cap([6000,8000]),[1800,2400]);np.testing.assert_allclose(cap([10,20]),[10,20])
    t=np.arange(18)*.02;true_acc=np.array([400.,-200.]);points=np.array([30.,40.])+t[:,None]*[80.,30.]+.5*t[:,None]**2*true_acc
    a,se,diag=robust_acceleration(list(zip(t,points)));assert diag['valid'];np.testing.assert_allclose(a,true_acc,atol=1e-5)
    corrupted=points.copy();corrupted[-2]+=[200,-150]
    a,se,diag=robust_acceleration(list(zip(t,corrupted)));d=np.column_stack([np.ones(len(t)),t,.5*t*t]);ols=np.linalg.lstsq(d,corrupted,rcond=None)[0][2]
    assert np.linalg.norm(a-true_acc)<np.linalg.norm(ols-true_acc)
    a,se,diag=robust_acceleration(list(zip(t[:6],points[:6])));assert not diag['valid']
    p=Predictor();q=Predictor();expected=[]
    for i in range(30):expected.append(p.step(i*.02,[100+i*2,70-i])[0])
    for i in range(15):assert q.step(i*.02,[100+i*2,70-i])[0]==expected[i]
    for row in expected:
        if row['valid']:assert np.linalg.norm(row['coefficients']['trusted_acc']['acceleration'])<=3000.+1e-8
    return dict(cap_norm_correct=True,clean_quadratic_acceleration_exact=True,isolated_outlier_less_influential_than_OLS=True,
        short_support_suppresses_acceleration=True,prefix_exact_ticks=15,trust_acc_cap_preserved=True,
        scope='Known synthetic geometry and causal invariants, not real-flight efficacy.')


if __name__=='__main__':print(run())
