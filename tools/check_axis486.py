"""Known-amplitude dense angular error propagation; no real accuracy claim."""
from pathlib import Path
import sys,numpy as np,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from shared_axis_error import SharedAxisMixin
from normal_response_observer import NormalResponseObserver

class Toy(NormalResponseObserver):
    state_dim=4;response_index=2
    def _transport_normal(self,n,scale):
        # Same local normal transport in a compact independent toy coordinate.
        h=self.x[2];self.x[:2]+=h*(scale*n-self.normal)
        J=np.eye(4);J[:2,2]=scale*n-self.normal;J[2,2]=scale
        self.P=J@self.P@J.T;self.x[2]*=scale
        tangent=np.array([-n[1],n[0]])
        self.P[:2,:2]+=(self.x[2]**2+self.P[2,2])*np.deg2rad(5)**2*np.outer(tangent,tangent)
        return False

# Direct independent scalar endpoint calculation shows the expected telescope.
if __name__=='__main__':
    h=1000.;variance=np.deg2rad(5)**2
    C=np.zeros((2,2)) # normal acceleration error and current angular error
    for i in range(30):
        J=np.array([[1.,-h],[0.,0.]]) if i else np.zeros((2,2))
        B=np.array([h,1.]);C=J@C@J.T+variance*np.outer(B,B)
        assert abs(C[0,0]-h*h*variance)<1e-7
    # Exercise the actual mixin with its runtime indices, holding h known.
    class Parent:
        def _transport_normal(self,n,scale):
            self.P[4:6,4:6]+=self.x[11]**2*variance*np.outer([-n[1],n[0]],[-n[1],n[0]])
            return False
    class Actual(SharedAxisMixin,Parent):state_dim=17;axis_error_index=16;response_index=11
    a=Actual();a.x=np.zeros(17);a.x[11]=h;a.P=np.zeros((17,17));a.normal=np.array([0.,-1.]);a.axis_error_active=False
    for _ in range(30):
        a._transport_normal(a.normal,1.)
        assert abs(a.P[4,4]-C[0,0])<1e-7
        assert np.linalg.eigvalsh(a.P).min()>-1e-7
    print(json.dumps(dict(shared_endpoint_variance=a.P[4,4],old_independent_variance_30frames=30*h*h*variance,
        fixed_normal_has_no_unbounded_orientation_noise_growth=True,real_accuracy_claim=False)))
