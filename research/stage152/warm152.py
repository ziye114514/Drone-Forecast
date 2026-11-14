"""Warm-start the unchanged fixed-shape objective using the previous solution.

Every fit still uses current full-conic perimeters, both shape hypotheses,
the same convergence tolerances, 80-iteration limit, and all existing gates.
The old solve remains the fallback when a recent solution is not a good start.
"""
from pathlib import Path
import ast,copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage138'),str(ROOT/'research/stage119')]
import objective138
import partial119
fixed117=objective138.fixed117

PARAMS=dict(maximum_initialization_age_s=.08,maximum_initial_rms_ratio=1.2,
    current_observations_unchanged=True,both_shape_hypotheses=True,
    optimization_tolerances_unchanged=True,default_promoted=False)

class Adapt(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if node.name=='fixed_fit':
            node.args.args.append(ast.arg(arg='warm_H'))
            node.args.defaults.append(ast.Constant(None))
        return self.generic_visit(node)
    def visit_Assign(self,node):
        node=self.generic_visit(node)
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='upper':
            return [node]+ast.parse('''
warm_used=False
if warm_H is not None:
    transform=np.array([[scale,0.,origin[0]],[0.,scale,origin[1]],[0.,0.,1.]])
    candidate=np.linalg.solve(transform,warm_H)
    candidate=candidate/candidate[2,2]
    candidate=candidate.reshape(-1)[:8]
    if np.isfinite(candidate).all():
        prior_rms=float(np.sqrt(np.mean(residual(candidate)**2)))
        cold_rms=float(np.sqrt(np.mean(residual(initial)**2)))
        if prior_rms<=max(1.,cold_rms*1.2):
            initial=candidate
            warm_used=True
''').body
        return node
    def visit_Return(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='dict' and any(k.arg=='projected_hubs' for k in node.value.keywords):
            node.value.keywords.append(ast.keyword(arg='warm_start152',value=ast.Name('warm_used',ast.Load())))
        return self.generic_visit(node)

tree=ast.fix_missing_locations(Adapt().visit(copy.deepcopy(objective138.tree)))
namespace=dict(objective138.namespace)
exec(compile(tree,'<152 previous-pose initialization of unchanged138 objective>','exec'),namespace)
warm_fixed_fit=namespace['fixed_fit']
OriginalFixedModel=fixed117.FixedModel

class WarmFixedModel(OriginalFixedModel):
    def __init__(self,guarded=False):
        super().__init__(guarded);self.pose_cache={};self.context_t=None;self.context_body=None
    def fit_current(self,rotors,shape):
        ellipses=[fixed117.geom.geometry58.from_rotor(z) for z in rotors]
        radius,aspect=shape;fits=[];used=0;iterations=0
        for dimensions in [(radius,aspect),(radius/aspect,1/aspect)]:
            key=tuple(float(v) for v in dimensions);old=self.pose_cache.get(key);prior=None
            if old is not None and self.context_t-old['t']<=PARAMS['maximum_initialization_age_s']+1e-9:
                prior=old['H'].copy()
                if self.context_body is not None and old['body'] is not None:
                    dx,dy=self.context_body-old['body'];prior=np.array([[1.,0.,dx],[0.,1.,dy],[0.,0.,1.]])@prior
            f=warm_fixed_fit(ellipses,fixed_shape=dimensions,warm_H=prior)
            if f.get('valid') and not f.get('optimizer_success') and f.get('warm_start152'):
                f=objective138.fixed_fit(ellipses,fixed_shape=dimensions)
                f['warm_fallback152']=True
            if f.get('valid') and f.get('optimizer_success') and f.get('same_projective_halfspace'):
                transform=np.array([[f['scale_px'],0.,f['origin_px'][0]],
                    [0.,f['scale_px'],f['origin_px'][1]],[0.,0.,1.]])
                self.pose_cache[key]=dict(t=self.context_t,H=transform@np.asarray(f['homography_normalized']),
                    body=None if self.context_body is None else self.context_body.copy())
            fits.append(f);used+=int(f.get('warm_start152',False));iterations+=f.get('nfev',0)
        good=[f for f in fits if f.get('valid') and f.get('optimizer_success') and f.get('same_projective_halfspace')]
        pool=good or [f for f in fits if f.get('valid')]
        if not pool:return dict(valid=False,reason='fixed_pose_optimization_failed',optimizer_success=False)
        best=min(pool,key=lambda f:f['rms_px']);best.update(shape_hypotheses_tested=2,
            warm_hypotheses152=used,total_nfev152=iterations)
        return best
    def step(self,t,body,rotors,axis,baseline):
        if self.last_body is not None and t-self.last_body>.5:self.__init__(self.guarded)
        self.context_t=t;self.context_body=np.asarray(body['center']) if body.get('detected') else None
        original=fixed117.fit_fixed;fixed117.fit_fixed=self.fit_current
        try:return super().step(t,body,rotors,axis,baseline)
        finally:fixed117.fit_fixed=original

def install():
    partial119.FixedModel=WarmFixedModel

def restore():
    partial119.FixedModel=OriginalFixedModel
