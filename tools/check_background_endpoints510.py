"""Known dense GLS solution for two shared image-error channels."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from background_relative_force import BackgroundReference

if __name__=='__main__':
    N=25;dt=1/30.;prior=1e6;noise=.5;rng=np.random.default_rng(510)
    fg=rng.normal(0,np.sqrt(2),N+1);bg=rng.normal(0,np.sqrt(2),N+1)
    y=dt*137.+np.diff(fg)-np.diff(bg)+rng.normal(0,np.sqrt(noise),N)
    C=np.eye(N)*(8+noise)+np.diag(np.full(N-1,-4.),1)+np.diag(np.full(N-1,-4.),-1)
    X=np.ones(N)*dt;precision=1/prior+X@np.linalg.solve(C,X)
    expected=(X@np.linalg.solve(C,y))/precision
    # Streaming conditional states: velocity, previous foreground/bg errors.
    x=np.zeros(3);P=np.diag([prior,2.,2.])
    for value in y:
        z=np.r_[x,0.,0.];V=np.zeros((5,5));V[:3,:3]=P;V[3:,3:]=np.eye(2)*2
        H=np.array([dt,-1.,1.,1.,-1.]);K=V@H/(H@V@H+noise)
        z+=K*(value-H@z);J=np.eye(5)-np.outer(K,H);V=J@V@J.T+noise*np.outer(K,K)
        ix=[0,3,4];x=z[ix];P=V[np.ix_(ix,ix)]
        assert np.linalg.eigvalsh(P).min()>-1e-8
    assert abs(x[0]-expected)<1e-7 and abs(P[0,0]-1/precision)<1e-7
    # Three directly observed camera pairs share two intermediate images.
    ref=BackgroundReference(shared_endpoints=True)
    for i in range(4):
        t=1+i*dt;ref.advance(dict(t=t,proof={},pair=None if i==0 else dict(
            source_t=t-dt,displacement_xy=[1.,0.],covariance_px2=(np.eye(2)*4).tolist())))
    body=dict(center=[5.,6.]);m=dict(dt_s=3*dt,estimate=dict(dx=4.,dy=0.,displacement_covariance_px2=np.eye(2).tolist()))
    b,m=ref.inputs(1+3*dt,body,m)
    assert m['background_reference_interval']['measured_intervals']==3
    assert np.allclose(m['estimate']['background_reference_added_covariance_px2'],np.eye(2)*4)
    assert np.allclose(m['estimate']['displacement_covariance_px2'],np.eye(2)*8)
    print(json.dumps(dict(streaming_velocity=float(x[0]),dense_GLS_velocity=float(expected),
        streaming_variance=float(P[0,0]),dense_GLS_variance=float(1/precision),
        known_three_pair_background_covariance=4.,wrong_independent_sum_covariance=12.,
        foreground_floor_preserved_before_adding_background=True,real_accuracy_claim=False)))
