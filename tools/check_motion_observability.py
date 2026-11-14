"""Small mechanism checks for new observation/response boundaries."""
from pathlib import Path
import copy,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion_observability import WeakSemanticMotion,visible_motion_box
from live_local_flow import box,reused_similarity
from shared_image_motion import SharedImageMotion
from adaptive_force_memory import simplex_fit,AdaptiveForceMemory,kernel
from tangential_force_response import fit_response
from shared_velocity_state import commit_velocity


def main():
    body=dict(detected=True,bbox=[100.,100.,180.,160.],center=[140.,130.],confidence=.3,low_confidence=True,
        measurement_source='current_KLT_from_delayed_NN',nn_observed_t=0.)
    tracking=dict(reason='current_image_translation',points=12,delta=[1.,0.],
        displacement_covariance_px2=np.eye(2).tolist(),similarity_motion=dict(valid=True,inliers=12,inlier_fraction=1.,
        displacement_covariance_px2=np.eye(2).tolist()))
    observer=WeakSemanticMotion()
    for i in range(12):
        d=observer.observe(i/30,body,tracking)
        assert not d['qualified']
    observer=WeakSemanticMotion()
    for i in range(7):
        b=dict(body,nn_observed_t=i/30)
        d=observer.observe(i/30,b,tracking)
    assert d['qualified'] and body['low_confidence'] and box(body) is None
    bad=copy.deepcopy(tracking);bad['similarity_motion']['displacement_covariance_px2']=(np.eye(2)*100).tolist()
    assert not observer.observe(8/30,dict(body,nn_observed_t=8/30),bad)['qualified']
    clipped=dict(body,low_confidence=False,bbox=[100.,-.6,180.,60.])
    assert box(clipped) is None and visible_motion_box(clipped) is not None
    assert visible_motion_box(dict(clipped,bbox=[100.,-20.,180.,60.])) is None
    center=np.array([140.,130.]);angles=np.arange(12)*2*np.pi/12
    points=center+np.column_stack((np.cos(angles)*30,np.sin(angles)*20))
    angle=np.radians(2);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    current=(points-center)@R.T+center
    sim=reused_similarity(points,current,np.zeros(12),center,spatial=True)
    assert sim['valid'] and np.linalg.norm(sim['displacement'])<1e-3
    assert sim['spatial_support']['center_in_hull']
    assert np.allclose(simplex_fit(np.eye(2),np.array([2.,-1.])),[1.,0.])
    assert np.allclose(simplex_fit(np.eye(2),np.array([1.,1.])),[.5,.5])
    memory=AdaptiveForceMemory(allow_shorter=True)
    force=dict(active=True,input_u=[0.,-1.],effective_gain_px_s2=200.,acceleration_px_s2=[1000.,0.],acceleration_correlation_s=.35)
    c=dict(position=[0.,0.],velocity=[10.,0.],acceleration=[1000.,0.])
    for i in range(30):
        t=i/30;c['position']=[10.*t,0.]
        pred={str(h):(np.array(c['position'])+h*np.array(c['velocity'])+kernel(-1/.35,h)*np.array([1000.,0.])).tolist() for h in (.1,.2,.4)}
        _,d=memory.step(t,c,c['position'],pred,force,current_total_qualified=True)
        if t<.2-1e-8:assert not d['updated'] and d['short_weight']==0.
    assert d['short_weight']>0. and d['short_weight']+d['weight']<=1.+1e-10
    rows=[dict(v=np.array([100.*np.exp(-2*t),0.]),a=np.array([-200.*np.exp(-2*t),0.]),se=np.array([.1,.1])) for t in np.linspace(0,.3,10)]
    beta,b,*_=fit_response(rows,np.array([1.,0.]));assert abs(beta-2.)<1e-8 and abs(b)<1e-8
    # Unknown cross-covariance bound, for a valid arbitrary joint covariance.
    rng=np.random.default_rng(1);A=rng.normal(size=(4,4));P=A@A.T+np.eye(4)
    bound=np.zeros((4,4));bound[:2,:2]=2*P[:2,:2];bound[2:,2:]=2*P[2:,2:]
    assert np.linalg.eigvalsh(bound-P).min()>-1e-10
    result=dict(duplicate_NN_not_multiple_anchors=True,weak_body_confidence_unchanged=True,
        high_displacement_uncertainty_rejected=True,slightly_truncated_motion_not_full_rotor_geometry=True,
        stationary_rotation_not_translation=True,force_memory_no_future_maturity=True,
        simplex_bounds=True,controlled_tangent_beta_recovery=True,unknown_joint_covariance_bound=True)
    path=ROOT.parent/'output/fred_transfer_20261029/mechanism_checks.json'
    path.write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main()
