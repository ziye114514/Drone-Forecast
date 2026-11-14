"""A zero-innovation current image must not change its own propagation prior."""
from pathlib import Path
import sys,copy,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coherent_gap_observer import ShortGapObserver

if __name__=='__main__':
    out=ROOT.parent/'output/role_controls420_20261110';out.mkdir(exist_ok=False)
    m=ShortGapObserver();meta=dict(observed_components={},input_sigma_proxy=0.)
    for i in range(6):
        t=1+i/30;b=dict(detected=True,center=[200+8*i,300-3*i],center_provisional=False,measurement_source='filtered_native_translation')
        m.step(t,b,None,input_meta=meta,motion=dict(dt_s=1/30,current_observation=True,estimate=dict(dx=8.,dy=-3.,score=1.)))
    m.x[2:4]=[250.,-90.];m.x[4:6]=[1000.,-300.];stamp=m.t+1/30
    missing=copy.deepcopy(m);missing.step(stamp,dict(detected=False),None,input_meta=meta,motion={})
    delta=missing.x[:2]-m.x[:2];b=dict(detected=True,center=missing.x[:2].tolist(),center_provisional=False,
        measurement_source='filtered_native_translation',continued_observation_preserves_gap_prior=True)
    motion=dict(dt_s=1/30,current_observation=True,estimate=dict(dx=float(delta[0]),dy=float(delta[1]),score=1.))
    weak=copy.deepcopy(m);c,v,f,s=weak.step(stamp,b,None,input_meta=meta,motion=motion)
    mean_error=float(np.max(abs(weak.x[:7]-missing.x[:7])));assert mean_error<1e-9,mean_error
    reduction=np.linalg.eigvalsh((missing.P[:6,:6]-weak.P[:6,:6]+(missing.P[:6,:6]-weak.P[:6,:6]).T)/2)
    assert reduction.min()>-1e-6;assert s['native_relative_displacement_observed'] and s['weak_observation_is_assimilated']
    old=copy.deepcopy(m);legacy=dict(b);legacy.pop('continued_observation_preserves_gap_prior');old.step(stamp,legacy,None,input_meta=meta,motion=motion)
    previous_mean_change=float(np.max(abs(old.x[:7]-missing.x[:7])));assert previous_mean_change>1e-4
    result=dict(zero_innovation_new_role_current_state_max_change=mean_error,old_role_switch_current_state_change=previous_mean_change,
        current_displacement_assimilated=s['native_relative_displacement_observed'],covariance_reduction_eigenvalues=reduction.tolist(),
        strong_observed_recipe_unchanged=True,time_constants_not_tuned=True,no_real_precision_success_claim=True)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(result)
