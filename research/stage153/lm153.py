"""Use MINPACK LM for the existing unbounded eight-parameter pose problem.

The original trust-region solver remains the fallback. This is an optimizer
change, not extra geometric evidence or a new learned motion prior.
"""
from pathlib import Path
import ast,copy,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage138')]
import objective138
fixed117=objective138.fixed117
PARAMS=dict(solver='LM for fixed eight-dimensional unbounded linear-loss objective',
    max_nfev_unchanged=80,observations_unchanged=True,both_hypotheses_unchanged=True,
    fallback='original TRF on unsuccessful or nonfinite LM result',default_promoted=False)
class Adapt(ast.NodeTransformer):
    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id=='least_squares':
            node.keywords.append(ast.keyword(arg='method',value=ast.Constant('lm')))
        return self.generic_visit(node)
tree=ast.fix_missing_locations(Adapt().visit(copy.deepcopy(objective138.tree)))
namespace=dict(objective138.namespace)
exec(compile(tree,'<153 MINPACK fixed8 objective>','exec'),namespace)
lm_fit=namespace['fixed_fit']
def fixed_fit(*args,**kwargs):
    f=lm_fit(*args,**kwargs)
    if not f.get('valid') or not f.get('optimizer_success') or not f.get('same_projective_halfspace'):
        f=objective138.fixed_fit(*args,**kwargs);f['lm_fallback153']=True
    else:f['lm_fallback153']=False
    return f
def install():fixed117.fixed_fit=fixed_fit
def restore():fixed117.fixed_fit=objective138.fixed_fit
