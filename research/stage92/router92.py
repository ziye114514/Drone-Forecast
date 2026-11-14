"""Reuse29 delayed expert router with a TRAIN-selected kinematic expert.

No detection, geometry, RPM, force parameters or old predictions are changed.
Existing primary is preserved as an explicit expert and reference.
"""
from pathlib import Path
from types import FunctionType
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/stage29'))
import select29

HISTORY=['cv','trusted_acc','kf_tuned']
FULL=['primary','cv','trusted_acc','kf_tuned']


def weighted_blend(coefficients,names,weights):
    c=select29.blend(coefficients,names,weights)
    # Original29 assumed same CV state for every expert. Here filters differ;
    # make metadata agree with the mixture, while preserving its26-point curve.
    c['position']=sum(w*np.asarray(select29.prior.evaluate(coefficients[m],0.)) for m,w in zip(names,weights)).tolist()
    c['velocity']=sum(w*np.asarray(coefficients[m]['velocity']) for m,w in zip(names,weights)).tolist()
    return c


def router(names):
    namespace=dict(select29.Selector.step.__globals__,HISTORY=names,EXPERTS=names,blend=weighted_blend)
    methods={name:FunctionType(getattr(select29.Selector,name).__code__,namespace,name) for name in ['reset','step']}
    return type('ReusedRouter29',(select29.Selector,),methods)()


class Router92:
    def __init__(self):self.history=router(HISTORY);self.full=router(FULL)
    def step(self,primary,kinematic,obs):
        q=dict(t=primary['t'],valid=primary['valid'],coefficients={},predictions={})
        if q['valid']:
            q['coefficients']={m:copy.deepcopy(kinematic['coefficients'][m]) for m in HISTORY}
            q['coefficients']['primary']=copy.deepcopy(primary['coefficients']['event'])
        h=self.history.step(q,obs);f=self.full.step(q,obs)
        coefficients={}
        if q['valid']:
            coefficients=dict(q['coefficients'])
            coefficients.update(history_router=h['coefficients']['adaptive_history_mix'],
                history_fixed=h['coefficients']['fixed_history_mix'],
                full_router=f['coefficients']['adaptive_history_mix'],full_fixed=f['coefficients']['fixed_history_mix'],
                fallback_router=copy.deepcopy(q['coefficients']['primary'] if primary['diagnostics']['fresh'] else h['coefficients']['adaptive_history_mix']))
        return dict(t=q['t'],valid=q['valid'],coefficients=coefficients,
            diagnostics=dict(fresh=primary['diagnostics']['fresh'],history=h['selection_diagnostic'],full=f['selection_diagnostic']))
