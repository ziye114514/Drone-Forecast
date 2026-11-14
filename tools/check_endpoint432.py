"""Independent dense conditioning of correlated difference measurements."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from shared_endpoint_observer import EndpointNoiseMixin
from relative_rotor_observer import RelativeRotorForceObserver


class ConstantVelocity:
    def _initialize(self,z):
        super()._initialize(z);self.P[4:7]=0.;self.P[:,4:7]=0.
    def _transition(self,dt):
        F=np.eye(self.state_dim);F[:2,2:4]=np.eye(2)*dt
        return F,np.zeros_like(F)


class Plain(ConstantVelocity,RelativeRotorForceObserver):pass
class Correlated(EndpointNoiseMixin,ConstantVelocity,RelativeRotorForceObserver):
    endpoint_base=9;state_dim=13


if __name__=='__main__':
    out=ROOT.parent/'output/endpoint_controls432_20261110';out.mkdir(exist_ok=False)
    dt=1/30;error=np.array([1.,-1.5,1.7,-.9,.4,1.3,-1.2,.5,-.8,.2,1.1,-.3])
    z=200+90*np.arange(len(error))*dt+error
    models=[Plain(),Correlated()];maximum=0.;covmax=0.;oldmax=0.
    for i,position in enumerate(z):
        b=dict(detected=True,center=[position,300.],center_provisional=False,
            measurement_source='filtered_native_translation' if i else 'rotor_region_initialization')
        motion=dict(current_observation=True,dt_s=dt,estimate=dict(dx=0. if i==0 else float(z[i]-z[i-1]),dy=0.,score=1.))
        for m in models:m.step(1+i*dt,b,None,input_meta=dict(observed_components={},input_sigma_proxy=0.),motion=motion)
        H=np.zeros((i+1,2));H[0,0]=1.;H[1:,1]=dt
        R=np.diag([36.]+[4.]*i)
        for k in range(i):R[k,k+1]=R[k+1,k]=-2.
        y=np.r_[z[0],np.diff(z[:i+1])];prior=np.array([z[0],0.]);precision=np.eye(2)/10000.
        C=np.linalg.inv(precision+H.T@np.linalg.solve(R,H));mean=C@(precision@prior+H.T@np.linalg.solve(R,y))
        T=np.array([[1.,i*dt],[0.,1.]]);mean=T@mean;C=T@C@T.T
        idx=[0,2];maximum=max(maximum,float(np.max(abs(models[1].x[idx]-mean))))
        covmax=max(covmax,float(np.max(abs(models[1].P[np.ix_(idx,idx)]-C))))
        oldmax=max(oldmax,float(np.max(abs(models[0].x[idx]-mean))))
    assert maximum<1e-7 and covmax<1e-7,(maximum,covmax)
    assert oldmax>1e-3
    result=dict(dense_correlated_Gaussian_mean_difference=maximum,dense_covariance_difference=covmax,
        independent_pair_control_difference=oldmax,variance_parameters_unchanged=True,real_prediction_success_claim=False)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(result)
