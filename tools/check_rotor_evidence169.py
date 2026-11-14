from pathlib import Path
import sys,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rotor_evidence import RotorEvidence,VisibilityLayout


if __name__=='__main__':
    points=[[100.,100.],[200.,100.],[200.,160.],[100.,160.]]
    disks=[dict(cx=x,cy=y,major=40.,minor=20.,angle=0.,ratio=.5,mode='ellipse',current_observation=True) for x,y in points]
    rates=[dict(valid=True,frequency_hz=f) for f in [270.,300.,330.,360.]]
    # Intentionally no initialized complete template or persistent motor IDs.
    s=dict(mode='insufficient',axis_xy=[0.,-1.],pose_source='current_qualified_visible_conics',inferred_model_ids=[0,1,2,3],observed_model_ids=[])
    b=dict(detected=True,center=[150.,130.]);model=RotorEvidence()
    model.step(0.,b,s,disks,rates);out=model.step(1/30,b,s,disks,rates)
    assert out['complete_current_frequency_sum'] and not out['template_confirmed']
    assert np.allclose(sum(np.array(v) for v in out['current_components'].values()),[0.,-sum(r['frequency_hz']**2 for r in rates)/360000.])
    perm=[2,0,3,1];other=model.step(2/30,b,s,[disks[i] for i in perm],[rates[i] for i in perm])
    assert other['current_components']==out['current_components']
    three=model.step(3/30,b,s,disks[:3],rates[:3]);assert three['current_contribution_count']==3
    assert all(a['frequency_hz'] is None and not a['measured'] for a in three['inferred_hubs'])
    missing=model.step(4/30,b,s,disks,[*rates[:3],dict(valid=False,frequency_hz=360.)])
    assert missing['period_region_count']==3 and not missing['complete_current_frequency_sum']
    line=[dict(r,minor=2.,ratio=.05,mode='line') for r in disks[:2]]
    side=model.step(5/30,b,s,line,rates[:2]);assert side['branch']=='side_axis' and side['current_contribution_count']==0
    e=RotorEvidence();layout=VisibilityLayout()
    for i in (0,1):layout.step(e.step(i/30,b,s,disks,rates))
    moved=[dict(r,cx=r['cx']+8,cy=r['cy']+1) for r in disks[:3]]
    g=layout.step(e.step(2/30,b,s,moved,rates[:3]))
    np.testing.assert_allclose(g['current_center'],[158.,131.],atol=1e-8)
    assert len(g['inferred_hubs'])==1 and g['inferred_hubs'][0]['frequency_hz'] is None
    g=layout.step(e.step(3/30,b,s,moved[:2],rates[:2]));assert g['current_center'] is None
    e=RotorEvidence();layout=VisibilityLayout()
    for i in (0,1):layout.step(e.step(i/30,b,s,disks,rates))
    tilted=[dict(r,cx=150+1.3*(r['cx']-150),major=52.,ratio=20/52) for r in disks[:3]]
    g=layout.step(e.step(2/30,b,s,tilted,rates[:3]))
    assert g['model']=='recent_layout_shape_checked_affine',g
    np.testing.assert_allclose(g['current_center'],[150.,130.],atol=1e-5)
    print('PASS: template-free measured4sum, permutation, partial3, no hidden RPM, invalid current period, side degeneracy')
