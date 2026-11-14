"""Known center motion under skewed feature support and rotation/scale."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from live_local_flow import reused_similarity
from rigid_center_translation import choose


def checks():
    center=np.array([600.,350.]);box=[520.,280.,680.,420.]
    # Surround the center, but put most corners on the right-hand rotor.
    points=center+np.vstack((np.array([[-60,-50],[-60,50],[60,-50],[60,50]]),
        np.column_stack((np.linspace(25,55,20),np.linspace(-30,20,20)))))
    theta=.06;A=1.015*np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    result={}
    for tag,translation in [('stationary_rotating_body',[0.,0.]),('translating_rotating_body',[4.,-3.])]:
        current=(points-center)@A.T+center+translation
        median=np.median(current-points,axis=0)
        sim=reused_similarity(points,current,np.zeros(len(points)),center,center_support=True)
        selected,info=choose(median,sim,box)
        assert info['applied'] and np.linalg.norm(selected-translation)<1e-3
        assert np.linalg.norm(median-translation)>1.
        result[tag]=dict(old_median=median.tolist(),new_center=selected.tolist(),truth=translation)
    sim['center_in_track_hull']=False
    selected,info=choose(median,sim,box);assert np.array_equal(selected,median) and not info['applied']
    selected,info=choose(median,dict(sim,center_in_track_hull=True,inliers=8),box)
    assert np.array_equal(selected,median) and not info['applied']
    result.update(one_sided_or_weak_fit_preserves_exact_fallback=True,new_optical_flow_passes=0)
    return result


if __name__=='__main__':print(json.dumps(checks(),indent=2))
