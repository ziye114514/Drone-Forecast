"""Observed-support joint fit, analytic Sampson Jacobian and local diagnostics.

Reuses the actual58 objective/initialization/optimizer via an inspected AST
adapter. Changes observations, optional standard robust loss and Jacobian.
Covariance is a local diagnostic; correlated morphology edges are not GT.
"""
from pathlib import Path
import ast,copy,inspect,sys,textwrap
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT/'research/stage58'),str(ROOT/'research/stage104')]
import geometry58,rigid104
from event_drone_forecast.geometry import _curved_points,_coverage

PARAMS=dict(max_points_per_rotor=64,robust_f_scale_px=1.5,noise_floor_px=.8,correlated_pixel_variance_factor=4.,
    minimum_support_coverage=.25,maximum_center_sigma_px=6.,shape_bound_log_tolerance=1e-4,
    modes=['perimeter_fd','perimeter_analytic','contour_linear','arc_robust','arc_guarded'],
    optimization='Same58 normalized H8+log(radius)+log(aspect), same bounds/max80 and initial H from four fitted centers. Analytic derivative of same pixel Sampson objective. No extra iterations, warm start or motion prior.',
    support='Actual selected threshold/morphology contours. Uniform index subsampling only among observed points. Arc variant reuses existing geometry._curved_points; never completes missing boundary.',
    uncertainty='SVD of full fit Jacobian; all H/radius/aspect parameters, floor.8px residual MAD, variance4 for pixel correlation. Uncalibrated local approximation; near-null rank explicitly invalid, bounds flagged rather than pseudo-observations.',
    guarded='Original104 admission plus no active/near shape bound, full-center sigma<=same6px limit, each observed arc angular coverage>=existing.25. Position/velocity/5-frame lock unchanged.',
    originality='Robust losses, analytic derivatives, ellipse uncertainty and arc selection have prior work. This adapter is a targeted research implementation, not an originality claim.')

def analytic_jacobian(p,points,template,scale):
    H=np.r_[p[:8],1.].reshape(3,3);A=np.linalg.inv(H);radius=np.exp(p[8]);aspect=np.exp(p[9])
    dA=[]
    for k in range(8):
        E=np.zeros((3,3));E.flat[k]=1.;dA.append(-A@E@A)
    parts=[]
    for xy,edge in zip(template,points):
        x,y=xy*np.array([1.,aspect]);C=np.array([[1.,0,-x],[0,1.,-y],[-x,-y,x*x+y*y-radius*radius]])
        P=A.T@C@A;cp=edge@P;n=np.sum(cp*edge,axis=1);length=np.linalg.norm(cp[:,:2],axis=1)
        den=np.maximum(2*length,1e-12)
        deriv=[d.T@C@A+A.T@C@d for d in dA]
        Cr=np.zeros((3,3));Cr[2,2]=-2*radius*radius;deriv.append(A.T@Cr@A)
        Ca=np.zeros((3,3));Ca[1,2]=Ca[2,1]=-y;Ca[2,2]=2*y*y;deriv.append(A.T@Ca@A)
        cols=[]
        for dP in deriv:
            dcp=edge@dP;dn=np.sum(dcp*edge,axis=1)
            dd=2*np.sum(cp[:,:2]*dcp[:,:2],axis=1)/np.maximum(length,1e-12)
            dd=np.where(2*length>1e-12,dd,0.)
            cols.append(scale*(dn/den-n*dd/den**2))
        parts.append(np.column_stack(cols))
    return np.concatenate(parts,axis=0)

def uncertainty(fit,scale):
    J=np.asarray(fit.jac);_,s,V=np.linalg.svd(J,full_matrices=False)
    rank=int(np.sum(s>max(s[0],1e-12)*1e-10));sigma=max(PARAMS['noise_floor_px'],1.4826*float(np.median(abs(fit.fun-np.median(fit.fun)))))
    bound=bool(np.any(np.asarray(fit.active_mask)[8:]!=0) or
        min(abs(fit.x[8]-np.log(.02)),abs(fit.x[8]-np.log(.99)),abs(fit.x[9]-np.log(.25)),abs(fit.x[9]-np.log(4.)))<PARAMS['shape_bound_log_tolerance'])
    if rank<10:return dict(rank=rank,center_sigma_px=None,center_covariance_px2=None,shape_on_bound=bound,calibrated=False,residual_sigma_px=sigma)
    covariance=(V.T*(1/s**2))@V*(sigma*sigma*PARAMS['correlated_pixel_variance_factor'])
    center=covariance[np.ix_([2,5],[2,5])]*scale**2
    return dict(rank=rank,center_sigma_px=float(np.sqrt(max(np.linalg.eigvalsh(center)[-1],0))),
        center_covariance_px2=center.tolist(),shape_on_bound=bound,residual_sigma_px=sigma,calibrated=False)

class Adapt(ast.NodeTransformer):
    def __init__(self):self.observed=0;self.optimizer=0;self.result=0
    def visit_FunctionDef(self,node):
        if node.name=='joint_fit':
            node.name='support_fit';node.args.args.extend([ast.arg(arg=n) for n in ['supports','fit_loss','use_analytic']])
            node.args.defaults.extend([ast.Constant(None),ast.Constant('linear'),ast.Constant(True)])
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='observed':
            self.observed+=1
            node.value=ast.IfExp(test=ast.Compare(left=ast.Name('supports',ast.Load()),ops=[ast.IsNot()],comparators=[ast.Constant(None)]),
                body=ast.parse('[(np.asarray(supports[i])-origin)/scale for i in order]',mode='eval').body,orelse=node.value)
        return self.generic_visit(node)
    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id=='least_squares':
            self.optimizer+=1
            for kw in node.keywords:
                if kw.arg=='loss':kw.value=ast.Name('fit_loss',ast.Load())
            node.keywords.extend([ast.keyword(arg='jac',value=ast.parse('(lambda p: analytic_jacobian(p,points,template,scale)) if use_analytic else "2-point"',mode='eval').body),
                ast.keyword(arg='f_scale',value=ast.Constant(PARAMS['robust_f_scale_px']))])
        return self.generic_visit(node)
    def visit_Return(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='dict' and any(k.arg=='projected_hubs' for k in node.value.keywords):
            self.result+=1
            for name,expr in [('uncertainty','uncertainty(fit,scale)'),('parameter_vector','fit.x.tolist()'),
                ('homography_normalized','H.tolist()'),('origin_px','origin.tolist()'),('scale_px','scale'),('rotor_order','order.tolist()')]:
                node.value.keywords.append(ast.keyword(arg=name,value=ast.parse(expr,mode='eval').body))
        return self.generic_visit(node)

source=textwrap.dedent(inspect.getsource(geometry58.joint_fit))
# Correct only the singular-inverse fallback length for unequal support sizes.
assert source.count('len(points)*len(points[0])')==1
source=source.replace('len(points)*len(points[0])','sum(len(v) for v in points)')
adapter=Adapt();tree=ast.fix_missing_locations(adapter.visit(ast.parse(source)))
assert (adapter.observed,adapter.optimizer,adapter.result)==(1,1,1)
DERIVED_SOURCE=ast.unparse(tree)
namespace=dict(geometry58.joint_fit.__globals__,analytic_jacobian=analytic_jacobian,uncertainty=uncertainty)
exec(compile(tree,'<113 observed support adapter>','exec'),namespace);support_fit=namespace['support_fit']

def supports_for(points,curved=False):
    result=[]
    for p in points:
        p=np.asarray(p,float).reshape(-1,2)
        if curved:p=_curved_points(p)
        if len(p)>PARAMS['max_points_per_rotor']:p=p[np.linspace(0,len(p)-1,PARAMS['max_points_per_rotor']).astype(int)]
        result.append(p)
    return result

def coverage(fit,supports):
    H=np.asarray(fit['homography_normalized']);inverse=np.linalg.inv(H);origin=np.asarray(fit['origin_px']);scale=fit['scale_px']
    aspect=fit['frame_aspect'];template=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])*[1.,aspect];values=[]
    for xy,i in zip(template,fit['rotor_order']):
        p=(np.asarray(supports[i])-origin)/scale;hp=np.c_[p,np.ones(len(p))]@inverse.T
        uv=hp[:,:2]/hp[:,2,None]-xy;angles=np.sort(np.arctan2(uv[:,1],uv[:,0])%(2*np.pi))
        values.append(float(1-np.max(np.diff(np.r_[angles,angles[0]+2*np.pi]))/(2*np.pi)))
    return values

measure_source=textwrap.dedent(inspect.getsource(rigid104.measure_four))
assert measure_source.count('fit=joint_fit([from_rotor(r) for r in current])')==1
measure_source=measure_source.replace('def measure_four(rotors,body):','def measure_fitted(rotors,body,provided_fit):').replace(
    'fit=joint_fit([from_rotor(r) for r in current])','fit=provided_fit')
ns=dict(rigid104.measure_four.__globals__);exec(compile(measure_source,'<113 original104 gates>','exec'),ns);measure_fitted=ns['measure_fitted']

step_source=textwrap.dedent(inspect.getsource(rigid104.PlanarCenter.step))
assert step_source.count('m=measure_four(rotors,body)')==1
step_source=step_source.replace('m=measure_four(rotors,body)','m=copy.deepcopy(self.input_measurement)')
ns2=dict(rigid104.PlanarCenter.step.__globals__);exec(compile(step_source,'<113 original104 model state>','exec'),ns2)
class MeasuredCenter(rigid104.PlanarCenter):step=ns2['step']

def guarded(m):
    m=copy.deepcopy(m)
    if not m['valid']:return m
    u=m['fit']['uncertainty'];m['full_center_uncertainty']=u
    if u['shape_on_bound']:return dict(m,valid=False,reason='shape_parameter_on_bound')
    if u['center_sigma_px'] is None or u['center_sigma_px']>PARAMS['maximum_center_sigma_px']:
        return dict(m,valid=False,reason='full_conic_center_uncertain')
    if min(m['fit']['support_coverage'])<PARAMS['minimum_support_coverage']:
        return dict(m,valid=False,reason='insufficient_observed_arc')
    m['center_noise_proxy_px']=max(m['center_noise_proxy_px'],u['center_sigma_px'])
    return m
