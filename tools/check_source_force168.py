from pathlib import Path
import sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from local_period_tracks import LocalPeriodTracks,InnovationUnknownForce
from rotor_thrust_bank import RotorThrustBankProxy
from rotor_force_observer import RotorForceObserver


def local(fs):return dict(regions=[dict(cell=[j,0],frequency_hz=f,other_supported_frequencies=[]) for j,f in enumerate(fs)])


if __name__=='__main__':
    model=LocalPeriodTracks();out=[]
    # Unequal speeds are not forced to one fleet RPM; dominance is irrelevant.
    for i in range(8):out.append(model.step(i/30,local([270.,330.,300.])))
    assert out[-1]['qualified'] and abs(out[-1]['relative_squared_proxy']-1)<1e-12
    out=model.step(8/30,local([324.,396.,360.]))
    assert out['qualified'] and abs(out['relative_squared_proxy']-1.44)<1e-10
    # Current-only alternatives can preserve one hypothesis; absent fundamental
    # is never manufactured by dividing the only observed harmonic by two.
    jump=model.step(9/30,local([648.,792.,720.]))
    assert not jump['qualified'] and jump['missing_fundamental_invented'] is False
    missing=model.step(10/30,dict(regions=[]));assert not missing['qualified']
    drift=LocalPeriodTracks();restarted=False
    for i in range(13):
        d=drift.step(i/30,local([60.*1.2**i]*3))
        restarted|=d['source_epoch']>0
    assert restarted
    for i in (13,14):d=drift.step(i/30,local([60.*1.2**12]*3))
    assert d['qualified'] and d['frequency_hz'] is not None
    q=InnovationUnknownForce();R=np.eye(2)*1e6;n=np.array([0.,-1.])
    a,_=q.step(0.,[0.,0.],{'a':1.},n,200.,R);assert np.allclose(a,0)
    a,info=q.step(1/30,[0.,0.],{'a':1.44},n,200.,R)
    assert a[1]<-1 and not np.allclose(a,0),a
    # Hiding/revealing source ownership alone keeps the current acceleration.
    q=InnovationUnknownForce();q.step(0.,[0.,0.],{'a':1.,'b':1.},n,200.,R)
    a,_=q.step(1/30,[0.,0.],{'a':1.},n,200.,R);assert np.allclose(a,0)
    a,_=q.step(2/30,[0.,0.],{'a':1.,'b':1.},n,200.,R);assert np.allclose(a,0)
    q=InnovationUnknownForce();a,_=q.step(0.,[0.,0.],{'a':1.},n,700.,R);assert np.allclose(a,0)
    for i in range(1,25):a,_=q.step(i/30,[500.,0.],{'a':1.},n,700.,R)
    assert abs(a[0]-500)<1 and abs(a[1])<1e-6,a
    bank=RotorThrustBankProxy(vector_pool=True)
    s=dict(mode='axis_only',axis_xy=[0.,-1.],geometry_epoch=0,geometric_weight=.8)
    for j,f in enumerate([300.,300.,360.]):
        u,q,meta=bank.step(j/30,s,[],[],dict(qualified=True,frequency_hz=f,
            selection_source='continued_local_regions:0',same_source_spatial_tracking_confirmed=True))
    assert u is not None and np.allclose(u,[0.,-.11])
    observer=RotorForceObserver();body=dict(detected=True,center=[100.,100.])
    for j in range(8):
        U=np.array([0.,0.]) if j<7 else np.array([0.,-.11])
        C,V,F,S=observer.step(j/30,body,U,.7,input_meta=dict(observed_components={'same':[float(U[0]),float(U[1])]},input_sigma_proxy=0.),allow_gain_learning=False)
    assert F['acceleration_px_s2'][1]<-1.,F
    print('PASS: unequal source speeds, current20percent response, alias rejection, missing evidence, balance/visibility, fast unknown update')
