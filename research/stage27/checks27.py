"""Meaningful invariants of the new same-state ablation harness."""
import copy
import numpy as np
from physics27 import Predictor,PHYSICS,evaluate


def sample(t,detected=True,features=True):
    rotors=[dict(cx=100.+i*20,cy=100.,major=30.,minor=24.,ratio=.8,angle=0.,geometry_quality=.95,activity=100.,observed_now=True) for i in range(4)] if features else []
    return dict(t=t,observation=dict(t=t,detected=detected,center=[100+30*t+20*t*t,120-10*t],geometry_quality=.95),
        sensing=dict(rotors=rotors,four_frequencies=features,four_visible=features,thrust_ratio=1+.2*np.sin(t) if features else None,frequency_quality=.95 if features else 0.))


def run():
    p=Predictor();records=[]
    for t in np.arange(.01,1.51,.01):
        q,b=p.step(sample(float(t)));records.append(q)
        if q['valid']:
            for m in b['coefficients']:assert q['coefficients'][m]==b['coefficients'][m]
            assert all(q['coefficients'][m]['position']==q['coefficients']['cv']['position'] and q['coefficients'][m]['velocity']==q['coefficients']['cv']['velocity'] for m in PHYSICS)
    assert records[-1]['feature_diagnostic']['models']['adaptive_axis_freq']['updates']>0
    prefix=Predictor()
    assert all(prefix.step(sample(float(t)))[0]==records[i] for i,t in enumerate(np.arange(.01,.81,.01)))
    for t in np.arange(1.51,1.81,.01):q,b=p.step(sample(float(t),False,False))
    assert not q['valid'] and not q['feature_diagnostic']['active']
    no=Predictor()
    for t in np.arange(.01,.31,.01):q,b=no.step(sample(float(t),True,False))
    assert q['valid'] and all(q['coefficients'][m]==q['coefficients']['cv'] for m in PHYSICS)
    c=dict(rollout_times=[0.,.2,.4],rollout_positions=[[0.,0.],[2.,4.],[4.,8.]])
    assert evaluate(c,.3)==[2.9999999999999996,5.999999999999999] or np.allclose(evaluate(c,.3),[3,6])
    try:evaluate(c,.6)
    except ValueError:pass
    else:raise AssertionError('rollout extrapolation accepted')
    return dict(same_cv_state_all_branches=True,baseline_unchanged=True,prefix_exact=True,
        adaptive_updates_after_delayed_observations=True,absent_features_exact_cv_fallback=True,dropout_expiry=True,saved_rollout_interpolation=True)
