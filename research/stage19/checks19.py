import copy,numpy as np
from layout19 import SoftGeometry,Observer,inc,counts


def rotor(x,y):
    return dict(center=np.array([x,y],float),bbox=np.array([x-12,y-5,x+12,y+5],float),major=24.,minor=10.,
        activity=100.,shape_quality=.9,geometry_quality=.9,normal=np.array([0.,.9]),angle=0.,ratio=.4,mode='ellipse')


def run():
    g=SoftGeometry(320,240,True,True);rr=[rotor(x,y) for x,y in [(60,70),(220,70),(220,170),(60,170)]]
    assert g._valid_four(rr);assert not g._valid_four([rotor(x,80) for x in [40,80,120,160]])
    g._remember(rr,1.);m=copy.deepcopy(g.memory)
    partial=[rotor(65,72),rotor(225,172),rotor(150,120)]
    fit=g._fit_memory(partial,1.02);assert fit and fit['matched']==2
    np.testing.assert_allclose(fit['points'],m['points']+[5,2],atol=1e-8)
    assert g._fit_memory(partial,1.32) is None
    assert g._fit_memory([rotor(60,70)],1.02) is None
    assert g._fit_memory([rotor(80,80),rotor(90,85)],1.02) is None
    # Disabled augmentation/layout exactly reproduces existing neighbor frontend.
    import cv2
    mask=np.zeros((240,320),np.uint8)
    for x,y in [(60,70),(220,70),(220,170),(60,170)]:cv2.ellipse(mask,(x,y),(22,7),12,0,360,1,-1)
    yy,xx=np.nonzero(mask);dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')])
    e=np.zeros(len(xx)*6,dtype);e['x']=np.repeat(xx,6);e['y']=np.repeat(yy,6);e['p']=np.tile([0,1,0,1,0,1],len(xx))
    base=inc.old.RecoveryFrontend(320,240,'neighbor10');a=Observer(320,240)
    spectrum=dict(mask=np.zeros((30,40),bool),hz=np.zeros((30,40)))
    for i in range(1,6):
        e['t']=i*20000-100;prepared=counts(e,320,240);prepared['on']/=2;prepared['off']/=2
        expected,_=base.detect_prepared(e,i*.02,prepared);q=a.step(e,i*.02,prepared,spectrum)
        assert q['observation']==expected
    return dict(partial_similarity_reuses_stage11=True,missing_slots_transform_correct=True,stale_full_template_rejected=True,
        single_and_collapsed_partial_rejected=True,degenerate_four_cannot_seed=True,baseline_neighbor_synthetic_exact_ticks=5,
        limitations='Synthetic geometry and adapter invariants, not real rotor identity/pose truth.')


if __name__=='__main__':print(run())
