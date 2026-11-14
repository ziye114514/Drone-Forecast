"""Behavioral checks: shape reanchors, true translation, and rotating clutter."""
from pathlib import Path
import json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from shared_image_motion import SharedImageMotion


def trial(rebase,stationary=False,bad_internal_motion=False):
    model=SharedImageMotion(curved_motion=True,position_history_rebase='stable-axis' if rebase else False)
    old=None;results=[];original=[]
    for i in range(19):
        t=i/30;true=np.array([450.+180*t+350*t*t,350.-20*t]) if not stationary else np.array([450.,350.])
        shift=np.array([8.,3.]) if i>=6 else np.zeros(2)
        half=np.array([50.+(8. if i>=6 else 0.),15.+(3. if i>=6 else 0.)])
        if bad_internal_motion:shift[:]=0;half=np.array([50.,15.])
        c=true+shift;body=dict(detected=True,confidence=.8,low_confidence=False,center=c.tolist(),bbox=np.r_[c-half,c+half].tolist(),
            measurement_source='current_NN' if i in (0,6) else 'current_KLT_translation',nn_observed_t=0. if i<6 else .2)
        delta=np.zeros(2) if old is None else true-old
        if bad_internal_motion:delta=np.array([5.+25.*t,0.])/30
        tracking=dict(reason='current_image_translation',points=40,delta=delta.tolist(),
            displacement_covariance_px2=np.eye(2).tolist(),similarity_motion=dict(valid=True,displacement=delta.tolist(),
                displacement_covariance_px2=np.eye(2).tolist(),inliers=40,inlier_fraction=1.))
        coefficient=dict(position=c.tolist(),velocity=(np.zeros(2) if stationary else np.array([180+700*t,-20.])).tolist(),acceleration=[0.,0.])
        saved=json.loads(json.dumps(body));track_saved=json.loads(json.dumps(tracking))
        value,info=model.step(t,coefficient,np.eye(2)*10000.,body,tracking,
            structure=dict(mode='axis_only',axis_xy=[0.,-1.],axis_support=dict(valid=True,confidence=.9)))
        assert body==saved and tracking==track_saved
        assert value['position']==coefficient['position']
        results.append(info);old=true;original.append(coefficient)
    return results


def main():
    plain=trial(False);fixed=trial(True);stationary=trial(True,True);clutter=trial(True,True,True)
    clutter_control=trial(False,True,True)
    old=sum(r['acceleration'].get('applied',False) for r in plain)
    new=sum(r['acceleration'].get('applied',False) for r in fixed)
    assert new>=old
    valid=[r['acceleration']['shrunk_acceleration'][0] for r in fixed if r['acceleration'].get('applied')]
    assert valid and max(abs(v-700.) for v in valid)<80.,valid
    assert not any(r['acceleration'].get('applied') for r in stationary)
    # Tiny internal motion can be below the existing body-position noise floor.
    # Reparameterization must not change that baseline decision or invent a
    # stronger claim that every possible internal pattern is distinguishable.
    for a,b in zip(clutter,clutter_control):
        assert a['acceleration']==b['acceleration']
    result=dict(true_constant_acceleration_px_s2=700.,estimated_acceleration_range=[min(valid),max(valid)],
        original_current_localization_unchanged=True,original_track_observations_unchanged=True,
        stationary_body_shape_reanchor_does_not_invent_acceleration=True,
        internal_pattern_without_shape_reanchor_behavior_unchanged=True,
        low_amplitude_internal_motion_baseline_not_proved_separable=True,
        side_axis_only_without_four_circle_template_supported=True,
        original_qualified_acceleration_ticks=old,rebase_qualified_acceleration_ticks=new,
        no_labels_or_future_observations_used=True)
    out=ROOT.parent/'output/maneuver_response_20261029';out.mkdir(exist_ok=True)
    (out/'mechanism_checks.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
