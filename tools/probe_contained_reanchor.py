"""A contained tracked patch is motion evidence, not a whole-body center.

Reproduce original math first. Evaluate the optional change on several frozen
observation streams with all source times and missing outputs preserved.
"""
from pathlib import Path
import copy,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from shared_image_motion import SharedImageMotion
from probe_same_window_motion import replay
from compare_final_pipeline import reference_at,stats
from score_fred_adaptation import matched
from diagnose_maneuver_horizon import SPECS
OPTIONS=dict(curved=True,evidence_gain=True,memory=True,endpoint=True,memory_gaps=True,
    weak_support=True,position_history_rebase='stable-axis')


def checks():
    old=dict(detected=True,low_confidence=True,confidence=.35,bbox=[470.,330.,530.,370.],
        center=[500.,350.],measurement_source='current_NN',nn_observed_t=0.)
    new=dict(detected=True,low_confidence=False,confidence=.8,bbox=[435.,320.,585.,380.],
        center=[510.,350.],measurement_source='current_KLT_from_delayed_NN',nn_observed_t=1/30)
    track=dict(reason='current_image_translation',points=14,delta=[-30.,0.],displacement_covariance_px2=np.eye(2).tolist(),
        similarity_motion=dict(valid=True,inliers=14,inlier_fraction=1.,displacement=[-30.,0.],
            displacement_covariance_px2=np.eye(2).tolist()))
    def run(enabled,body=None,tracking=None):
        m=SharedImageMotion(curved_motion=True,contained_reanchor_motion=enabled)
        m.previous=(0.,copy.deepcopy(old));b=copy.deepcopy(new if body is None else body)
        before=copy.deepcopy(b);c=dict(position=b['center'],velocity=[0.,0.],acceleration=[0.,0.])
        _,d=m.step(1/30,c,np.eye(2)*1e5,b,track if tracking is None else tracking)
        assert b==before,'motion qualification mutated the current box'
        if d['observation'].get('contained_weak_patch_motion'):
            assert not d['acceleration']['applied'] and len(m.derivative.velocities)==0
        return d['observation']
    control=run(False);candidate=run(True)
    assert not control['valid'] and candidate['valid']
    assert np.allclose(candidate['velocity'],[-900.,0.])
    assert candidate['derivative_eligible'] is False
    from flow_force_balance import FlowBalancedForce
    f=FlowBalancedForce();f.observe_flow(candidate,False)
    f.step(1/30,dict(position=new['center'],velocity=[-900.,0.],acceleration=[0.,0.]),new['center'],None,.35)
    assert len(f.flow_velocities)==0
    for bad in [dict(track,points=8),dict(track,similarity_motion=dict(track['similarity_motion'],inliers=8)),
        dict(track,similarity_motion=dict(track['similarity_motion'],inlier_fraction=.6)),
        dict(track,similarity_motion=dict(track['similarity_motion'],displacement_covariance_px2=(np.eye(2)*16).tolist()))]:
        assert not run(True,tracking=bad)['valid']
    assert not run(True,body=dict(new,bbox=[700.,320.,850.,380.],center=[775.,350.]))['valid']
    assert not run(True,body=dict(new,low_confidence=True,confidence=.35))['valid']
    assert not run(True,body=dict(new,measurement_source='current_KLT_translation',nn_observed_t=0.))['valid']
    return dict(original_rejects_partial_expansion=True,contained_patch_displacement_not_center_jump=True,
        current_body_unchanged=True,weak_or_uncontained_or_unsupported_or_old_anchor_rejected=True,
        changing_reference_not_added_to_acceleration_or_force_derivative_history=True,
        current_motion_is_not_measured_rotor_thrust=True)


def main(out):
    out.mkdir(parents=True,exist_ok=False);base=ROOT.parent/'output'
    sources=[('side_search','side',base/'transport_stability_20261029/RAW_side_search8s/predictions.jsonl'),
        ('other_search','other',base/'transport_stability_20261029/RAW_other_search_observed6s/predictions.jsonl'),
        ('side_previous','side',base/'maneuver_response_20261029/RAW_side_axis_rebase8s/predictions.jsonl'),
        ('other_previous','other',base/'maneuver_response_20261029/RAW_other_axis_rebase6s/predictions.jsonl'),
        ('side_frozen','side',base/'maneuver_response_20261029/side_stable_side_rebase.jsonl'),
        ('other_frozen','other',base/'maneuver_response_20261029/other_stable_side_rebase.jsonl')]
    result=dict(checks=checks(),streams={},RAW_FPS_measured=False,parameters_swept=False)
    for key,name,path in sources:
        rows=[json.loads(x) for x in path.read_text('utf-8').splitlines()]
        control,_=replay(rows,False,**OPTIONS)
        difference=max((float(np.max(np.abs(np.asarray(a['physical'][h])-b['physical'][h])))
            for a,b in zip(rows,control) for h in a['physical']),default=0.)
        assert difference<1e-7,(key,difference)
        candidate,_=replay(rows,False,contained_reanchor_motion=True,**OPTIONS)
        assert all(a['valid']==b['valid'] and a['body']==b['body'] for a,b in zip(rows,candidate))
        _,original,label=next(s for s in SPECS if s[0]==name)
        labels=sorted(json.loads((base/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        pair=matched(rows,candidate,lambda t:reference_at(labels,t))
        original_rows=[json.loads(x) for x in (base/original).read_text().splitlines()]
        original_pair=matched(original_rows,candidate,lambda t:reference_at(labels,t))
        accepted=[r['t'] for r in candidate if r['image_motion']['observation'].get('contained_weak_patch_motion')]
        (out/(key+'_patched.jsonl')).write_text('\n'.join(json.dumps(x,allow_nan=False) for x in candidate)+'\n','utf-8')
        result['streams'][key]=dict(original_reproduced_max_difference=difference,matched_same_observations=pair,
            fixed_original_origins=original_pair,accepted_current_patch_motion_times=accepted)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({k:dict(old=v['matched_same_observations']['old_matched_error'],
        new=v['matched_same_observations']['new_matched_error'],accepted=v['accepted_current_patch_motion_times'],
        fixed=v['fixed_original_origins']['new_matched_error']) for k,v in result['streams'].items()},indent=2))


if __name__=='__main__':main(Path(sys.argv[1]))
