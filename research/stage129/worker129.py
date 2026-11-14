"""Add FOV guard to the actual128 worker in the SAME process.

Only startup + before-current-response insertions; original worker128 source
is retained/hash-frozen. No extra raw decoder or heavyweight child process.
"""
from pathlib import Path
import argparse,ast,inspect,sys,textwrap
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage128'),str(ROOT/'research/stage129')]
import worker128
from boundary129 import Guarded
class AddGuard(ast.NodeTransformer):
    def __init__(self):self.changed=[]
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='counter':
            self.changed.append('startup');return [node]+ast.parse("safe_model=Guarded(ready['width'],ready['height'])").body
        return self.generic_visit(node)
    def visit_keyword(self,node):
        if node.arg=='stage' and isinstance(node.value,ast.Constant) and node.value.value==128:
            node.value=ast.Constant(129);self.changed.append('header')
        return self.generic_visit(node)
    def visit_Expr(self,node):
        call=node.value
        if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='print' and call.args:
            inner=call.args[0]
            if isinstance(inner,ast.Call) and inner.args and isinstance(inner.args[0],ast.Name) and inner.args[0].id=='result':
                self.changed.append('response')
                return ast.parse("""safe_result=safe_model.step(t,request['body'],request['axis'],result['input_rotors'],occupancy)
safe_coefficient=safe_result['forecast']['coefficients'].get(NAMES[2])
if safe_coefficient is not None:
    result['coefficients']['support_FOV_guard_kf129']=safe_coefficient
result['support129']=safe_result
""").body+[node]
        return self.generic_visit(node)
tree=ast.parse(textwrap.dedent(inspect.getsource(worker128.main)));adapter=AddGuard();tree=ast.fix_missing_locations(adapter.visit(tree));assert adapter.changed==['startup','header','response'],adapter.changed
namespace=dict(worker128.main.__globals__,Guarded=Guarded);exec(compile(tree,'<129 FOV insertion in128 worker>','exec'),namespace);main=namespace['main']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--duration',type=float,required=True);a=p.parse_args();main(a.raw,a.start,a.duration)
