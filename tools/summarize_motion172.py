"""Post-run fault report. Labels are used here only, never by the front end."""
from pathlib import Path
import sys, json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'src')]
from relative_rotor_observer import RelativeRotorForceObserver
from score_rotor164 import score

OUT=ROOT.parent/'output/motion172_20261107'

def mechanism():
    m=RelativeRotorForceObserver()
    for i in range(6):
        t=i/30
        m.step(t,dict(detected=True,center=[100+i,200],measurement_source='filtered_native_translation'),None,
               motion=dict(dt_s=None if i==0 else 1/30,estimate=dict(dx=1,dy=0,score=1,peak_at_search_boundary=False)))
    seen=m.seen
    c,valid,f,s=m.step(6/30,dict(detected=True,center=[10000,10000],center_provisional=True,
        measurement_source='predicted_rotor_reference_after_rejected_match',
        motion_reference_advanced_without_measurement=True),None,
        motion=dict(dt_s=1/30,estimate=dict(dx=40,dy=0,score=1,peak_at_search_boundary=True)))
    assert m.seen==seen
    assert np.linalg.norm(c['position'])<1000 and s['measurement_innovation_chi2'] is None
    assert s['prediction_only_image_reference_cloned'] and not s['native_relative_displacement_observed']
    c2,v2,f2,s2=m.step(7/30,dict(detected=True,center=[10000,10000],measurement_source='filtered_native_translation'),None,
        motion=dict(dt_s=1/30,estimate=dict(dx=1,dy=0,score=1,peak_at_search_boundary=False)))
    assert s2['native_relative_displacement_observed'] and np.linalg.norm(c2['position'])<1000
    assert np.linalg.eigvalsh(m.P).min()>-1e-6
    from unittest.mock import patch
    from types import SimpleNamespace
    from rotor_only_front import RotorOnlyFront
    front=RotorOnlyFront(guard_invalid_motion=True)
    on=np.ones((720,1280),np.float32);mask=np.zeros(on.shape,bool);mask[190:210,90:110]=True
    front.previous=(on,on);front.previous_mask=mask;front.last_t=0.;front.center=np.array([100.,200.])
    front.center_confirmed=True;front.last_velocity=np.array([15.,2.]);seen_velocity=[]
    def geometry_step(*args,**kwargs):
        seen_velocity.append(args[4].copy())
        return dict(mode='edge_on',current_geometric_center=None),None,{},None
    front.geometry=SimpleNamespace(step=geometry_step)
    front.provide_phase_events=False
    frame=dict(on=on,off=on,native_precomputed=(mask,{}),period_mask=mask,
               counts=dict(period_pixels=400))
    events=np.zeros(0,dtype=[('t','i8'),('x','u2'),('y','u2'),('p','i2')])
    with patch('rotor_only_front.rotor_support',return_value=frame),patch('rotor_only_front.estimate_region_translation',
        return_value=dict(dx=40.,dy=0.,score=1.,peak_at_search_boundary=True)):
        result=front.step(events,1/30)
    np.testing.assert_allclose(seen_velocity[0],[15.,2.])
    np.testing.assert_allclose(result['body']['center'],[100.5,200+2/30])
    assert result['body']['center_provisional'] and not result['motion']['current_observation']
    return dict(bad_center_not_observed=True,seen_not_advanced=True,
        prediction_reference_cloned=True,next_current_relative_measurement_used=True,
        rejected_match_not_geometry_velocity=True,predicted_reference_not_current_position_measurement=True,
        covariance_min_eigenvalue=float(np.linalg.eigvalsh(m.P).min()),
        bad_input_center=[10000,10000],position_after_bad=c['position'],position_after_good=c2['position'],
        synthetic_fault_mechanism_not_real_accuracy=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out-name',default='motion_guard172');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT/'guard_mechanism.json').write_text(json.dumps(mechanism(),indent=2),'utf-8')
    folder=ROOT.parent/'output/geometry170_20261106'/args.out_name
    assert (folder/'completion.json').exists()
    labels=ROOT.parent/'output/dense_labels_train2_40_46/labels.json'
    result={}
    for arm in ('legacy','fixed'):
        path=folder/f'train2_{arm}.jsonl'
        result[arm]=score(path,labels,np.arange(43.2,44.4001,.2))
        rows=[json.loads(s) for s in path.read_text('utf-8').splitlines()]
        result[arm]['rejected_matches']=[dict(t=r['t'],body=r['body'],motion=r['motion'],state=r['state']) for r in rows
            if ((r.get('motion') or {}).get('estimate') or {}).get('peak_at_search_boundary',False)]
    (OUT/(args.out_name+'_scores.json')).write_text(json.dumps(result,indent=2),'utf-8')
    for arm,r in result.items():
        print(arm,dict(available=r['available'],error=r['error'],
            rows=[{k:q.get(k) for k in ('t','error_px','current_position_error_px')} for q in r['records']],
            rejected=[p['t'] for p in r['rejected_matches']]))
