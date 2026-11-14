import copy
import numpy as np
from select29 import Selector,EXPERTS,ADDED,prior


def stream():
    result=[]
    for i in range(1,151):
        t=i*.01;c={m:dict(position=[t*50,100.],velocity=[50.,0.],acceleration=[0.,0.]) for m in prior.MODELS}
        # Deliberately poor non-CV experts make the expected direction testable.
        for m in EXPERTS[1:]:c[m]['velocity']=[-300.,0.]
        q=dict(t=t,valid=True,last_measurement_age_s=0.,real_observations=i,anchor_has_current_measurement=True,coefficients=c,predictions={})
        result.append((q,dict(t=t,detected=True,confidence=.9,center=[t*50,100.])))
    return result


def run():
    data=stream();p=Selector();out=[]
    for q,o in data:
        r=p.step(q,o);out.append(r)
        assert all(r['coefficients'][m]==q['coefficients'][m] for m in prior.MODELS)
        assert all(abs(sum(w.values())-1)<1e-10 and min(w.values())>0 for w in r['selection_diagnostic']['weights'].values())
        if q['t']<.21-1e-8:assert r['selection_diagnostic']['updates']==0
    assert out[-1]['selection_diagnostic']['weights']['all']['cv']>.9
    prefix=Selector();assert all(prefix.step(q,o)==r for (q,o),r in zip(data[:60],out[:60]))
    q,o=copy.deepcopy(data[-1]);q['t']=2.1;q['valid']=False;q['coefficients']={};q['predictions']={};o['detected']=False
    r=p.step(q,o);assert not r['valid'] and r['selection_diagnostic']['updates']==0
    return dict(original_coefficients_unchanged=True,weights_positive_normalized=True,no_feedback_before_delay=True,
        predictable_expert_gains_weight=True,prefix_exact=True,dropout_resets_and_does_not_publish=True)
