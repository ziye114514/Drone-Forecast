import numpy as np
from axis28 import Predictor,METHODS,CanonicalStatic,prior
from checks27 import sample


def run():
    for method in METHODS:
        p=Predictor(method);old=prior.Predictor();records=[]
        for t in np.arange(.01,.91,.01):
            r=sample(float(t));q,b=p.step(r);original,_=old.step(r);records.append(q)
            if method=='original':assert q==original
            for name in prior.robust.MODELS:assert q['coefficients'].get(name)==original['coefficients'].get(name)
        pref=Predictor(method)
        assert all(pref.step(sample(float(t)))[0]==records[i] for i,t in enumerate(np.arange(.01,.51,.01)))
    for instant in [False,True]:
        f=CanonicalStatic(prior.PaperConfig(),instant)
        for i,angle in enumerate([0,89,179,1,91,179,0]*8):
            r=sample((i+1)*.01)
            for rotor in r['sensing']['rotors']:rotor.update(angle=angle,ratio=.08)
            n=f.step(r['t'],r['sensing']['rotors']);assert np.isclose(np.linalg.norm(n),1) and n[1]<=1e-10 and n[2]>=-1e-10
        before=f.n.copy();f.step(1.,[]);assert np.array_equal(before,f.n) and f.last_measurement_t<1.
    return dict(original_exact=True,all_history_baselines_unchanged=True,prefix_all_variants_exact=True,
        canonical_signed_halfspace_invariant=True,empty_input_does_not_refresh_axis_age=True,unit_normal=True)
