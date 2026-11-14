"""Batch fixed-shape residual and Jacobian; share values at the same iterate."""
from pathlib import Path
import ast,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage137')]
import fast137
fixed117=fast137.fixed117

class FixedObjective:
    def __init__(self,points,template,scale,shape):
        self.edge=np.asarray(points);self.scale=scale;self.p=None;self.J=None
        radius,aspect=shape;xy=np.asarray(template)*[1.,aspect];x,y=xy.T
        self.C=np.zeros((len(x),3,3));self.C[:,0,0]=self.C[:,1,1]=1.
        self.C[:,0,2]=self.C[:,2,0]=-x;self.C[:,1,2]=self.C[:,2,1]=-y;self.C[:,2,2]=x*x+y*y-radius*radius
        self.rr,self.cc=np.unravel_index(np.arange(8),(3,3))
    def compute(self,p):
        if self.p is not None and np.array_equal(p,self.p):return
        A=np.linalg.inv(np.r_[p,1.].reshape(3,3));CA=self.C@A;P=A.T@CA;cp=np.einsum('rni,rij->rnj',self.edge,P)
        n=np.sum(cp*self.edge,axis=-1);length=np.linalg.norm(cp[...,:2],axis=-1);den=np.maximum(2*length,1e-12)
        self.A=A;self.CA=CA;self.cp=cp;self.n=n;self.length=length;self.den=den;self.value=(self.scale*n/den).reshape(-1);self.p=np.array(p,copy=True);self.J=None
    def fun(self,p):
        try:self.compute(p)
        except np.linalg.LinAlgError:return np.full(self.edge.shape[0]*self.edge.shape[1],1e6)
        return self.value
    def jac(self,p):
        self.compute(p)
        if self.J is not None:return self.J
        A=self.A;dA=-A[:,self.rr].T[:,:,None]*A[self.cc,:][:,None,:]
        dP=dA.transpose(0,2,1)[None]@self.CA[:,None]+(A.T@self.C)[:,None]@dA[None]
        dcp=np.einsum('rni,rkij->rnkj',self.edge,dP);dn=np.sum(dcp*self.edge[:,:,None,:],axis=-1)
        dd=2*np.sum(self.cp[:,:,None,:2]*dcp[...,:2],axis=-1)/np.maximum(self.length[:,:,None],1e-12);dd=np.where(2*self.length[:,:,None]>1e-12,dd,0.)
        self.J=(self.scale*(dn/self.den[:,:,None]-self.n[:,:,None]*dd/self.den[:,:,None]**2)).reshape(-1,8)
        return self.J

class Adapter(ast.NodeTransformer):
    def __init__(self):self.changed=[]
    def visit_FunctionDef(self,node):
        if node.name=='residual':
            self.changed.append('residual');node.body=ast.parse('return evaluation.fun(p)').body
            return ast.parse('evaluation=FixedObjective(points,template,scale,fixed_shape)').body+[node]
        return self.generic_visit(node)
    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id=='least_squares':
            for kw in node.keywords:
                if kw.arg=='jac':kw.value=ast.parse('evaluation.jac',mode='eval').body;self.changed.append('jacobian')
        return self.generic_visit(node)
adapter=Adapter();tree=ast.fix_missing_locations(adapter.visit(ast.parse(fixed117.DERIVED_FIXED_SOURCE)));assert adapter.changed==['residual','jacobian']
namespace=dict(fixed117.namespace,FixedObjective=FixedObjective);exec(compile(tree,'<138 batched cached fixed objective>','exec'),namespace);fixed_fit=namespace['fixed_fit']
def install():fixed117.fixed_fit=fixed_fit
