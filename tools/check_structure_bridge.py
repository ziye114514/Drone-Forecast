"""Mechanism controls for recent-model reuse, count-independent axes and CI."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from five_circle_model import FiveCircleModel,collinear_axis
from live_structure_bridge import QualifiedCenter,V7StructurePair
from check_five_circle_v3 import sequence
from structure_activity import WholeBodyPeriodicity
from whole_body_axis import measure


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    model=FiveCircleModel(3);rows=[model.step(t,b,g) for t,b,g in sequence()]
    assert rows[45]['mode']=='four' and rows[45]['recent_model_reused']
    cold=FiveCircleModel(3);t,b,g=next(sequence())
    assert cold.step(t,b,g)['mode']=='relock_pending'
    #3current regions support an axis even when motor labels are ambiguous.
    rotors=[dict(cx=100+40*i,cy=200+4*i,major=30.,minor=3.,angle=5.7,ratio=.1,
                 current_observation=True,association_ambiguous=i==2) for i in range(3)]
    assert collinear_axis(rotors)['valid']
    for r in rotors:r['ratio']=1.;r['minor']=r['major']
    assert not collinear_axis(rotors)['valid']
    centre=QualifiedCenter();body=dict(detected=True,confidence=.8,center=[640.,360.],bbox=[600.,320.,680.,400.])
    geometry=dict(geometry_epoch=0,mode='four',current_geometric_center=[640.,360.],
                  joint_center_is_current_measurement=True,geometric_weight=.8)
    p,a=centre.step(body,geometry,[]);assert np.allclose(p['center'],body['center']) and a['noise_scale']>=1.
    bad=dict(geometry,current_geometric_center=[1000.,1000.])
    p,a=centre.step(body,bad,[]);assert p==body and not a['applied']
    p,a=centre.step(dict(body,detected=False),geometry,[]);assert not a['applied'] and not p['detected']
    pair=V7StructurePair();empty=dict(geometry_epoch=0,mode='insufficient',current_geometric_center=None,
                                  joint_center_is_current_measurement=False)
    frame=np.full((720,1280,3),52,np.uint8);updates=published=0
    for i in range(100):
        t=i*.02;b=dict(body,center=[640+20*t,360.])
        result,pred=pair.step(t,b,empty,[],frame)
        if result['updated']:
            updates+=1;published+=result['published_source_slot']
            assert result['coefficients']['v7_same_observations']==result['coefficients']['v7_structure']
            assert result['valid']['v7_same_observations']==result['valid']['v7_structure']
    assert updates==50 and published==10
    activity=WholeBodyPeriodicity()
    rates=[dict(valid=False,instantaneous146=dict(valid=True,frequency_hz=f,pixels=20,mode_fraction=.9)) for f in [280.,300.]]
    assert not activity.step(0.,rates,0)['valid']
    q=activity.step(.02,rates,0);assert q['valid'] and q['hidden_periods_added']==0 and not q['mechanical_RPM_measured']
    assert not activity.step(.04,[dict(r,past_only_frequency_held=True) for r in rates],0)['valid']
    assert not activity.step(.06,rates,1)['valid']
    native=np.full((720,1280,3),52,np.uint8)
    for y in range(350,357):
        native[y,600:681:2,0]=255;native[y,601:681:2,0]=200
    assert measure(native,body)['valid']
    counts=np.zeros((720,1280),np.float32);counts[350:357,600:681]=5.
    assert measure(native,body,dict(on=counts,off=counts))['valid']
    native[:]=52
    yy,xx=np.mgrid[:720,:1280];disk=(xx-640)**2+(yy-360)**2<20**2
    native[disk,0]=255;native[disk & (xx%2==0),0]=200
    assert not measure(native,body)['valid']
    counts[:]=0.;counts[disk]=5.
    assert not measure(native,body,dict(on=counts,off=counts))['valid']
    #Exercise the inherited dynamic reset path with a long observation gap.
    for i in range(101,140):pair.step(i*.02,dict(body,detected=False,center=None),empty,[],frame)
    result,_=pair.step(2.84,body,empty,[],frame)
    assert result['updated']
    summary=dict(recent_compatible_relock=True,cold_start_still_confirmed=True,
        three_region_axis=True,round_region_negative=True,geometry_jump_rejected=True,
        missing_body_not_manufactured=True,no_geometry_branches_numerically_equal=True,
        exact25Hz_source_updates=updates,exact5Hz_source_slots=published,
        gap_reset_passed=True,RAW_started=False,parameters_swept=False)
    (out/'checks.json').write_text(json.dumps(summary,indent=2),'utf-8');print(json.dumps(summary))


if __name__=='__main__':run(sys.argv[1])
