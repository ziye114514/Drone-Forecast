"""Reference invariants, baseline parity and source-offset cancellation."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src')]
from coherent_gap_observer import ShortGapObserver
from layout_reference_observer import LayoutReferenceObserver,NativeControlLayoutObserver
from physical_publication import evaluate
OUT=ROOT.parent/'output/layout176_20261107'

def run(offset,geometry=True):
    m=LayoutReferenceObserver();base=ShortGapObserver();max_first_delta=0.;rows=[];max_no_geo=0.
    for i in range(30):
        t=round(i*1e6/30)/1e6;p=np.array([100.+90*t,200.-20*t]);delta=np.array([3.,-2/3]) if i else np.zeros(2)
        body=dict(detected=True,center=p.tolist(),measurement_source='filtered_native_translation',confidence=.7)
        motion=dict(dt_s=None if i==0 else t-round((i-1)*1e6/30)/1e6,
                    estimate=dict(dx=float(delta[0]),dy=float(delta[1]),score=1.,peak_at_search_boundary=False))
        # Native integer cutoffs introduce subpixel interval variation.
        if i:motion['estimate'].update(dx=90*motion['dt_s'],dy=-20*motion['dt_s'])
        if geometry and i in (12,15,18,21,24,27):
            motion['layout_observation']=dict(center=(p+np.array([8.,-24.])+offset).tolist(),
                covariance_proxy=(36*np.eye(2)).tolist(),epoch=0,current_observation=True)
        c,v,f,s=m.step(t,body,None,input_meta={},motion=motion)
        cb,vb,fb,sb=base.step(t,body,None,input_meta={},motion=motion)
        if i<=12 or not geometry:
            for key in ('position','velocity','acceleration'):
                diff=float(np.max(abs(np.array(c[key])-cb[key])));max_no_geo=max(max_no_geo,diff)
            if i==12:max_first_delta=max_no_geo
        assert np.linalg.eigvalsh(m.P).min()>-1e-5
        rows.append(dict(t=t,coefficient=c,force=f,state=s))
    assert max_no_geo<1e-8,max_no_geo
    return rows,m,dict(first_geometry_no_position_velocity_force_change=max_first_delta,no_geometry_parity=max_no_geo)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    a,ma,checks=run(np.zeros(2));b,mb,_=run(np.array([70.,-110.]));unused,_,_=run(np.zeros(2),False)
    maximum=0.
    for x,y in zip(a,b):
        for key in ('position','velocity','acceleration'):
            maximum=max(maximum,float(np.max(abs(np.array(x['coefficient'][key])-y['coefficient'][key]))))
    assert maximum<1e-8,maximum
    H=np.zeros((2,11));H[:,:2]=np.eye(2);H[:,9:11]=np.eye(2)
    before=H@ma.x;v=ma.x[2:7].copy();ma._rebase_observation_reference(np.array([11.,-7.]))
    np.testing.assert_allclose(H@ma.x,before,atol=1e-10);np.testing.assert_array_equal(ma.x[2:7],v)
    assert np.linalg.eigvalsh(ma.P).min()>-1e-5
    checks.update(arbitrary_layout_coordinate_offset_cancels=maximum,
        support_reference_correction_keeps_p_plus_bias=True,velocity_force_unchanged_by_rebase=True,
        layout_updates=sum(r['state']['layout_observation_used'] for r in a),
        covariance_min_eigenvalue=float(np.linalg.eigvalsh(ma.P).min()),synthetic_mechanism_not_forecast_accuracy=True)
    assert checks['layout_updates']>0
    (OUT/'mechanism.json').write_text(json.dumps(checks,indent=2),'utf-8');print(json.dumps(checks))
