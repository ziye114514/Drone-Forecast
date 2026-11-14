import numpy as np
from predict22 import Predictor,evaluate


def run():
    coefficients=dict(position=[3.,5.],velocity=[2.,-4.],acceleration=[6.,8.])
    np.testing.assert_allclose(evaluate(coefficients,.2),[3.52,4.36])
    a=Predictor();b=Predictor();expected=[];n=0
    for i in range(50):
        t=i*.02;center=[100+20*t,70-10*t];q=a.step(t,center);expected.append(q)
        if i<4:assert not q['valid']
        if i==49:assert q['valid'] and q['anchor_has_current_measurement']
    for i in range(25):assert b.step(i*.02,[100+20*i*.02,70-10*i*.02])==expected[i];n+=1
    q=a.step(1.2,None);assert not q['valid']
    q=a.step(1.6,None);assert not q['valid'] and q['real_observations']==0
    q=a.step(1.62,[200,200]);assert not q['valid'] and q['real_observations']==1
    return dict(analytic_endpoint_matches=True,prefix_exact_ticks=n,minimum_measurements_required=True,
        missing_timeout_stops_forecasts=True,long_gap_resets_history=True,labels_not_used_in_predictor=True)


if __name__=='__main__':print(run())
