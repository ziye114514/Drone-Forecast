"""Same geometry objective; batch its Jacobian and share existing support masks.

No ROI crop or changed threshold/fit/initialization/qualification. Reuses124's
merge-mask cache also for the count arm. Occupancy has a constant threshold,
so the full-image positive quantile was unused and is removed there only.
"""
from pathlib import Path
import ast,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage124'),str(ROOT/'research/stage117')]
import occupancy124,polarity,fixed117
from occupancy124 import SupportFrontend,MergeRecovery
PARAMS=dict(geometry_objective='unchanged pixel Sampson distance and optimizer',
    jacobian='Batch same analytic derivative across rotors and8 homography coordinates. Fixed structure has no radius/aspect derivative; original117 computed those then discarded.',
    count_support='Existing124 count_global uses exactly the original global count threshold and caches its raw mask for merge protection; no recomputed Gaussian/quantile.',
    occupancy_support='Existing constant0.5 support threshold; omit unused global positive-bin extraction and quantile.',
    qualification='Unchanged129 full-boundary guard, original five-frame model lock, partial affine3 and all model gates.',
    ROI='Do not crop contours or alter global threshold; current body ROI retains original centroid-only gate.')

def fast_fixed_jac(p,points,template,scale,fixed_shape):
    H=np.r_[p[:8],1.].reshape(3,3);A=np.linalg.inv(H)
    radius,aspect=fixed_shape;edge=np.asarray(points)
    xy=np.asarray(template)*[1.,aspect];x,y=xy.T;n=len(x)
    C=np.zeros((n,3,3));C[:,0,0]=C[:,1,1]=1.
    C[:,0,2]=C[:,2,0]=-x;C[:,1,2]=C[:,2,1]=-y
    C[:,2,2]=x*x+y*y-radius*radius
    rr,cc=np.unravel_index(np.arange(8),(3,3))
    dA=-A[:,rr].T[:,:,None]*A[cc,:][:,None,:]
    CA=C@A;AC=A.T@C;P=A.T@CA
    dP=dA.transpose(0,2,1)[None]@CA[:,None]+AC[:,None]@dA[None]
    cp=np.einsum('rni,rij->rnj',edge,P)
    dcp=np.einsum('rni,rkij->rnkj',edge,dP)
    numerator=np.sum(cp*edge,axis=-1);length=np.linalg.norm(cp[...,:2],axis=-1);den=np.maximum(2*length,1e-12)
    dn=np.sum(dcp*edge[:,:,None,:],axis=-1)
    dd=2*np.sum(cp[:,:,None,:2]*dcp[...,:2],axis=-1)/np.maximum(length[:,:,None],1e-12)
    dd=np.where(2*length[:,:,None]>1e-12,dd,0.)
    return (scale*(dn/den[:,:,None]-numerator[:,:,None]*dd/den[:,:,None]**2)).reshape(-1,8)

class RemoveUnused(ast.NodeTransformer):
    def __init__(self):self.changed=[]
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ['positive_bins','threshold']:
                self.changed.append(name);node.value=ast.Constant(None if name=='positive_bins' else .5)
        return self.generic_visit(node)
tree=ast.parse(ast.unparse(occupancy124.tree));adapter=RemoveUnused();tree=ast.fix_missing_locations(adapter.visit(tree));assert adapter.changed==['positive_bins','threshold']
namespace=dict(occupancy124.ns);exec(compile(tree,'<137 omit unused occupancy quantile>','exec'),namespace);occupancy_detect=namespace['detect']

class FastOccupancy(SupportFrontend):
    def __init__(self,w,h):super().__init__(w,h,'occupancy_roi')
    def detect_prepared(self,events,t,prepared):
        original=polarity._reused_detect;polarity._reused_detect=occupancy_detect
        try:return MergeRecovery.detect_prepared(self,events,t,prepared)
        finally:polarity._reused_detect=original

def setup(w,h):
    fixed117.fixed_fit.__globals__['fixed_jac']=fast_fixed_jac
    return dict(count=SupportFrontend(w,h,'count_global'),occupancy=FastOccupancy(w,h))
