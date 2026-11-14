"""Avoid an unused free-shape optimization after structure is learned.

Original117 immediately replaces the result of free H/radius/aspect fitting
with a fixed-shape fit whenever free fitting is reached. Preserve every
pre-fit admission check; supply a marker to run the same fixed fit directly.
Before structure learning or after identity expiry, original free fit remains.
"""
from pathlib import Path
import ast,inspect,sys,textwrap
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage113')]
import support113 as geom
class Preflight(ast.NodeTransformer):
    def __init__(self):self.changed=0
    def visit_FunctionDef(self,node):node.name='preflight';return self.generic_visit(node)
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='fit' and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='joint_fit':
            self.changed+=1;return ast.parse('return dict(diag, fit={}, free_fit_skipped=True)').body
        return self.generic_visit(node)
tree=ast.parse(textwrap.dedent(inspect.getsource(geom.rigid104.measure_four)));adapter=Preflight();tree=ast.fix_missing_locations(adapter.visit(tree));assert adapter.changed==1
namespace=dict(geom.rigid104.measure_four.__globals__);exec(compile(tree,'<136 unchanged104 prefit checks>','exec'),namespace);preflight=namespace['preflight']
def baseline_for(model,t,rotors,body):
    fixed=model.model
    if fixed.shape is None or (fixed.last_body is not None and t-fixed.last_body>.5):return geom.rigid104.measure_four(rotors,body)
    return preflight(rotors,body)
