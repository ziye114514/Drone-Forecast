"""Learn dimensionless structure at existing five-frame lock, then fit pose only.

Structure memory is distinct from a fresh pose, position state and identity.
This is model-based estimation using standard optimization, not a novelty claim.
"""
from pathlib import Path
import ast,copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage113')]
import support113 as geom
PARAMS=dict(modes=['free_structure','fixed_structure','fixed_guarded'],
    calibration='Exactly existing104 qualified five-frame lock. Canonicalize90deg template relabel, median log common-radius and frame aspect. Additional canonical aspect spread<=existing1.35 factor. No metric dimensions or manual shape.',
    persistence='Keep learned structure while the same body identity has current support within.5s. Old geometry freshness/layout/anchor rules and5-frame reacquisition remain; missing pose is never a new measurement.',
    optimization='H8 only after calibration, two90deg template-order hypotheses selected by current residual among successful same-halfspace fits. Same80 budget per hypothesis and3px/other104 gates. All current full-conic samples unchanged, no extra sensor/GT.',
    covariance='Pose-conditional Jacobian covariance only; excludes uncertainty/bias of five-frame learned structure, explicitly uncalibrated. Guarded uses existing6px threshold; no claim of calibrated full uncertainty.',
    maximum_canonical_aspect_spread=1.35)

def fixed_jac(p,points,template,scale,fixed_shape):
    return geom.analytic_jacobian(np.r_[p,np.log(fixed_shape)],points,template,scale)[:,:8]

def pose_uncertainty(fit,scale):
    _,s,V=np.linalg.svd(fit.jac,full_matrices=False);rank=int(np.sum(s>max(s[0],1e-12)*1e-10))
    sigma=max(.8,1.4826*float(np.median(abs(fit.fun-np.median(fit.fun)))))
    if rank<8:return dict(rank=rank,center_sigma_px=None,conditional_on_fixed_structure=True,calibrated=False)
    cov=(V.T*(1/s**2))@V*(sigma*sigma*4);cc=cov[np.ix_([2,5],[2,5])]*scale**2
    return dict(rank=rank,center_sigma_px=float(np.sqrt(max(np.linalg.eigvalsh(cc)[-1],0))),
        center_covariance_px2=cc.tolist(),conditional_on_fixed_structure=True,structure_uncertainty_included=False,calibrated=False)

class Fix(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if node.name=='support_fit':
            node.name='fixed_fit';node.args.args.append(ast.arg(arg='fixed_shape'));node.args.defaults.append(ast.Constant(None))
        elif node.name=='unpack':
            node.body=ast.parse('return np.r_[p[:8],1.].reshape(3,3), fixed_shape[0], fixed_shape[1]').body
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['initial','lower','upper']:
            node.value=ast.Subscript(value=node.value,slice=ast.Slice(upper=ast.Constant(8)),ctx=ast.Load())
        return self.generic_visit(node)
    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id=='analytic_jacobian':
            node.func.id='fixed_jac';node.args.append(ast.Name('fixed_shape',ast.Load()))
        elif isinstance(node.func,ast.Name) and node.func.id=='uncertainty':node.func.id='pose_uncertainty'
        return self.generic_visit(node)

tree=ast.fix_missing_locations(Fix().visit(ast.parse(geom.DERIVED_SOURCE)));DERIVED_FIXED_SOURCE=ast.unparse(tree)
namespace=dict(geom.namespace,fixed_jac=fixed_jac,pose_uncertainty=pose_uncertainty)
exec(compile(tree,'<117 fixed learned structure>','exec'),namespace);fixed_fit=namespace['fixed_fit']

def canonical(radius,aspect):
    return np.array([radius,aspect]) if aspect>=1 else np.array([radius/aspect,1/aspect])

def fit_fixed(rotors,shape):
    ellipses=[geom.geometry58.from_rotor(z) for z in rotors];radius,aspect=shape
    fits=[fixed_fit(ellipses,fixed_shape=(radius,aspect)),fixed_fit(ellipses,fixed_shape=(radius/aspect,1/aspect))]
    good=[f for f in fits if f.get('valid') and f.get('optimizer_success') and f.get('same_projective_halfspace')]
    pool=good or [f for f in fits if f.get('valid')]
    if not pool:return dict(valid=False,reason='fixed_pose_optimization_failed',optimizer_success=False)
    best=min(pool,key=lambda f:f['rms_px']);best['shape_hypotheses_tested']=2;return best

class FixedModel:
    def __init__(self,guarded=False):
        self.center=geom.MeasuredCenter();self.shape=None;self.shape_samples=None;self.last_body=None;self.guarded=guarded;self.learned_at=None
    def step(self,t,body,rotors,axis,baseline):
        if self.last_body is not None and t-self.last_body>.5:self.__init__(self.guarded)
        if body.get('detected'):self.last_body=t
        m=copy.deepcopy(baseline);fixed_used=False
        if self.shape is not None and baseline.get('fit') is not None:
            fit=fit_fixed(rotors,self.shape);m=geom.measure_fitted(rotors,body,fit);fixed_used=True
            if self.guarded and m['valid']:
                sigma=fit['uncertainty']['center_sigma_px']
                if sigma is None or sigma>6:m=dict(m,valid=False,reason='fixed_pose_center_uncertain')
                else:m['center_noise_proxy_px']=max(m['center_noise_proxy_px'],sigma)
        self.center.input_measurement=m;obs,d=self.center.step(t,body,rotors,axis);learned=False
        if self.shape is None and d['initialized_now']:
            samples=np.array([canonical(q['radius'],q['frame_aspect']) for _,q in self.center.pending])
            if samples[:,1].max()/samples[:,1].min()<=PARAMS['maximum_canonical_aspect_spread']:
                self.shape=np.exp(np.median(np.log(samples),axis=0));self.shape_samples=samples.tolist();self.learned_at=t;learned=True
        d.update(structure_ready=self.shape is not None,structure_learned_now=learned,structure_learned_at=self.learned_at,
            structure=None if self.shape is None else self.shape.tolist(),structure_initial_samples=self.shape_samples if learned else None,
            fixed_structure_used=fixed_used,structure_is_metric=False,structure_memory_is_current_pose=False,
            guarded_pose_covariance_excludes_learned_shape_uncertainty=True)
        return obs,d
