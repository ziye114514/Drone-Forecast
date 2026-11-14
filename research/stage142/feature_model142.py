"""Experimental pixel acceleration model from141 periodic line evidence.

Reuses the actual ProjectedThrustModel arithmetic as a feature regressor.
Rename its old four-visible gate to feature-supported explicitly: no four
rotors, true RPM or actual aggregate thrust are asserted by this adapter.
"""
from pathlib import Path
import ast,copy,inspect,sys,textwrap
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src')]
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast import paper_physics as original
PARAMS=dict(source='Actual existing ProjectedThrustModel, unchanged RLS delay.2s, warmup.6s, excitation.035, bounds, residual update and forgetting. Gate identifier renamed to feature_supported.',
    feature='Current141 qualified line, dominant flicker frequency squared in first fixed reference, normal sign assumed from image-up. Not all motor speeds, actual thrust or observed normal sign.',
    admission='Current140 dense/unclipped body fit, no full rotor structure,141 current line/flicker, >=10 model updates, update age<=.26s. Existing.2..5 frequency-ratio range.',
    baseline='140 state position/velocity retained; replace future acceleration only when admitted. No state feedback.',
    experimental='One fixed reuse pilot;141/140 development outcomes already known. No claim parameters identify mass, wind, cf or gravity uniquely.')
class Rename(ast.NodeTransformer):
    def visit_ClassDef(self,node):node.name='AxisFeatureModel';return self.generic_visit(node)
    def visit_Name(self,node):
        if node.id=='four_visible':node.id='feature_supported'
        return node
    def visit_arg(self,node):
        if node.arg=='four_visible':node.arg='feature_supported'
        return node
tree=ast.fix_missing_locations(Rename().visit(ast.parse(textwrap.dedent(inspect.getsource(original.ProjectedThrustModel)))));namespace=dict(original.__dict__);exec(compile(tree,'<142 unchanged RLS, feature support semantics>','exec'),namespace);AxisFeatureModel=namespace['AxisFeatureModel']

class Model:
    def __init__(self):self.model=AxisFeatureModel(PaperConfig(),adaptive=True);self.reference=None
    def step(self,r):
        coefficient=r['coefficients']['optimized_geometry'];c=copy.deepcopy(coefficient);q=r.get('periodic_axis') or {};gate=r['predictor'].get('dense_motion_fallback',{});n=None;s=1.;quality=0.
        supported=bool(q.get('qualified') and q.get('axis_valid') and gate.get('eligible') and gate.get('unclipped_fit_window') and not r['predictor']['structure_supported'])
        if supported:
            f=q['frequency_hz']
            if self.reference is None:self.reference=f
            s=(f/self.reference)**2;n=np.r_[q['line']['normal'],0.]
            if n[:2]@np.array([0.,-1.])<0:n=-n
            quality=q['line']['geometry_quality'];supported=.2<=s<=5.
        if c is not None:
            state=np.r_[c['position'],c['velocity']];self.model.update(r['t'],state,n,s,quality,r['body']['detected'],supported)
        active=bool(supported and c is not None and self.model.updates>=10 and self.model.last_update is not None and r['t']-self.model.last_update<=.26)
        if active:c['acceleration']=self.model.acceleration(n,s,np.asarray(c['velocity'])).tolist()
        return dict(t=r['t'],coefficient=c,active=active,supported=supported,updates=self.model.updates,parameters=self.model.parameters.tolist(),reference_flicker_hz=self.reference,relative_rate_squared=s if supported else None,normal_sign_assumed=True,actual_mechanical_RPM_or_thrust=False)
