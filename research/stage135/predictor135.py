"""Deployable optimized forecast: relative geometry velocity + robust rollout.

Use the richer rollout only after an independent guarded rotor model has
actually learned this target. Unmodeled/unsupported detections keep the body
CV branch. Learned structure and current geometric motion remain separate.
"""
from pathlib import Path
import copy,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage132')]
from fast132 import Fast,forecast
PARAMS=dict(rule='Before either guarded model learns structure: body CV. After guarded structure exists for the same current body identity: relative-geometry velocity + unchanged23 uncertainty-shrunk robust acceleration rollout. Body/velocity state is not anchored to first conic position.',
    geometry='Only current129 FOV-qualified model centers update velocity; constant source offset cancels. Old model memory is not a new measurement.',
    rich_rollout='Use original132 robust acceleration rollout, not a six-state acceleration feedback loop. No changed acceleration/noise thresholds.',
    identity='Models and predictor retain original.5s body-gap reset. No geometry learned in a new identity means plain body-CV fallback.',
    ablation='Same structure-memory condition with body CV velocity vs relative-geometry velocity. This separates identity gating from event motion contribution.',
    history='130/131/132/133/134 development outcomes known; no blind-test claim. Conditional rule fixed before135 measurement. No per-video rule or future error used at runtime.')
class Predictor:
    def __init__(self):self.fast=Fast()
    def step(self,t,body,models):
        result=self.fast.step(t,body,models);learned=any(d.get('structure_ready',False) for d in models.values());coeff={}
        for name,key in [('optimized_geometry','fast_geometry_acc'),('matched_body','fast_body_acc')]:
            c=copy.deepcopy(result[key]['coefficient'])
            if c is not None and not learned:c['acceleration']=[0.,0.]
            coeff[name]=c
        return dict(coefficients=coeff,predictions={name:{str(h):forecast(c,h) for h in [.1,.2,.4]} for name,c in coeff.items() if c is not None},
            structure_supported=learned,current_geometry_velocity_source=result['fast_geometry_acc']['source'],
            acceleration=result['acceleration_diagnostic'],no_absolute_rotor_position_anchor=True)
