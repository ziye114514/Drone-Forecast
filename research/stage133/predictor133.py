"""Fast relative-geometry motion + robust acceleration with reused70 selection.

Position/velocity are never permanently anchored to the first rotor model.
Acceleration fit resets across a>40ms observation gap. The actual unchanged70
Guard class selects using predictions that have already matured against the
current detector, not sparse future annotations.
"""
from pathlib import Path
from collections import deque
from types import SimpleNamespace
import ast,copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage132')]
from fast132 import Fast,forecast
ADDED=['body_acc','geometry_acc','body_acc_constant','geometry_acc_constant']
source=ast.parse((ROOT/'research/stage70/guard70.py').read_text('utf-8'));node=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='Guard')
namespace=dict(deque=deque,copy=copy,np=np,prior=SimpleNamespace(evaluate=forecast),ADDED=ADDED)
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<133 original70 Guard class>','exec'),namespace);Guard=namespace['Guard']
PARAMS=dict(acceleration_history_reset_gap_s=.04000001,minimum_observations=5,maximum_coast_s=.16,
    motion='132 fast107 body/relative rotor velocity; unchanged23 robust acceleration used only for rollout.',
    selection='Actual70 Guard class, .2/.4s matured detector feedback, .6s rolling median, >=5 feedbacks,5% advantage at both horizons,>=10 updates. Same class arithmetic, mapped experts. All coefficients generated online before later measurements.',
    base='geometry-relative CV, identical body CV without current rotor observations; no first-frame absolute rotor anchor.',
    experts=ADDED,feedback='Current detector center is a proxy, not independent truth.',default_promoted=False)
class Predictor:
    def __init__(self):self.fast=Fast();self.guard=Guard();self.last_body=None;self.observations=0;self.updates={m:0 for m in ADDED}
    def step(self,t,body,models):
        if self.last_body is not None and t-self.last_body>.5:self.__init__()
        if body.get('detected'):
            if self.last_body is not None and t-self.last_body>PARAMS['acceleration_history_reset_gap_s']:self.fast.history.clear()
            self.last_body=t;self.observations+=1
        result=self.fast.step(t,body,models);b=result['fast_body_acc']['coefficient'];g=result['fast_geometry_acc']['coefficient']
        coeff={};active={};fit_valid=result['acceleration_diagnostic']['valid'];geo=result['fast_geometry_acc']['source'] is not None
        if g is not None:
            for name,value in [('trusted_acc',g),('body_acc',b),('geometry_acc',g),('body_acc_constant',b),('geometry_acc_constant',g)]:
                c=copy.deepcopy(value)
                if name=='trusted_acc' or name.endswith('_constant'):c['acceleration']=[0.,0.]
                coeff[name]=c
            for name in ADDED:
                a=bool(body.get('detected') and (geo if name.startswith('geometry') else True) and (fit_valid if not name.endswith('_constant') else True));self.updates[name]+=a;active[name]=dict(active=a,updates=self.updates[name])
        valid=bool(g is not None and self.observations>=5 and self.last_body is not None and t-self.last_body<=.16000001)
        q=dict(t=t,valid=valid,coefficients=coeff,predictions={m:{str(h):forecast(c,h) for h in [.1,.2,.4]} for m,c in coeff.items()},event_force_diagnostic=dict(models=active))
        out=self.guard.step(q,body);choice=out['force_guard_diagnostic']['choices']['guarded_all']['chosen'];selected=out['coefficients'].get('guarded_all') if valid else None
        return dict(t=t,valid=valid,coefficient=selected,predictions={str(h):forecast(selected,h) for h in [.1,.2,.4]} if selected else {},chosen=choice,
            baseline_coefficient=coeff.get('trusted_acc'),body_coefficient=coeff.get('body_acc_constant'),fast=result,selection=out['force_guard_diagnostic'],named_coefficients=coeff)
