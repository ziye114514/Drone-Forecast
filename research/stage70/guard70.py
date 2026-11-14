"""Delayed observed-error guard for the event-force experiment (standard selection).

Admission is based on previously matured predictions, never the future test
boxes. Event models additionally need to beat their matched constant control.
This protects a useful history estimate without claiming that safety proves
the event signal carries predictive information.
"""
from collections import deque
from pathlib import Path
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/stage69'))
from force69 import ADDED,MODELS as PREVIOUS_MODELS,prior
ADDED70=['guarded_events','guarded_constants','guarded_all']
MODELS=PREVIOUS_MODELS+ADDED70
PARAMS=dict(delays=[.2,.4],maximum_extra_age_s=.06,recent_feedback_s=.6,
    minimum_feedback=5,minimum_relative_gain=.05,minimum_model_updates=10,
    observation_confidence_min=.5,base='trusted_acc',
    scope='Rolling median prediction error against current detector positions. '
          'Require past advantage at both .2 and .4s; events also beat matched '
          'constant input. Same future-label coverage. This gate is standard '
          'online model selection, not proof of a new thrust law.')


class Guard:
    def __init__(self):self.reset()
    def reset(self):
        self.queue={h:deque() for h in [.2,.4]};self.errors={h:deque() for h in [.2,.4]};self.last_seen=None
    def step(self,forecast,obs):
        q=copy.deepcopy(forecast);t=q['t']
        if self.last_seen is not None and t-self.last_seen>.5:self.reset()
        observed=bool(obs.get('detected') and obs.get('confidence',0)>.5)
        if obs.get('detected'):self.last_seen=t
        for h in [.2,.4]:
            while self.queue[h] and t-self.queue[h][0][0]>h+.06000001:self.queue[h].popleft()
            if observed:
                while self.queue[h] and t-self.queue[h][0][0]>=h-1e-8:
                    ot,old=self.queue[h].popleft();dt=t-ot;z=np.asarray(obs['center'])
                    errors={m:float(np.linalg.norm(np.asarray(prior.evaluate(c,dt))-z)) for m,c in old.items()}
                    self.errors[h].append((t,errors))
            while self.errors[h] and t-self.errors[h][0][0]>.6+1e-8:self.errors[h].popleft()
        loss={h:{m:float(np.median([e[m] for _,e in self.errors[h]])) for m in ['trusted_acc']+ADDED} for h in [.2,.4] if len(self.errors[h])>=5}
        diagnostics={};md=q['event_force_diagnostic']['models']
        pools=dict(guarded_events=[m for m in ADDED if not m.endswith('_constant')],
                   guarded_constants=[m for m in ADDED if m.endswith('_constant')],guarded_all=ADDED)
        for name,pool in pools.items():
            qualified=[]
            if len(loss)==2:
                for candidate in pool:
                    if not md[candidate]['active'] or md[candidate]['updates']<10:continue
                    control=candidate+'_constant' if not candidate.endswith('_constant') else None
                    comparisons=['trusted_acc']+([control] if name=='guarded_events' else [])
                    if all(loss[h][candidate]<.95*loss[h][other] for h in [.2,.4] for other in comparisons):
                        qualified.append(candidate)
            choice=min(qualified,key=lambda m:loss[.4][m]) if qualified else 'trusted_acc'
            if q['valid']:
                q['coefficients'][name]=copy.deepcopy(q['coefficients'][choice])
                q['predictions'][name]=copy.deepcopy(q['predictions'][choice])
            diagnostics[name]=dict(chosen=choice,qualified=qualified)
        if q['valid']:
            for h in [.2,.4]:self.queue[h].append((t,{m:copy.deepcopy(q['coefficients'][m]) for m in ['trusted_acc']+ADDED}))
        q['force_guard_diagnostic']=dict(feedback_counts={str(h):len(self.errors[h]) for h in [.2,.4]},
            losses={str(h):d for h,d in loss.items()},choices=diagnostics)
        return q


def controls():
    # Every physical candidate is deliberately worse in already matured
    # predictions. A strong history forecast must then survive exactly.
    g=Guard()
    for j in range(80):
        t=j*.02;base=dict(position=[t,0.],velocity=[1.,0.],acceleration=[0.,0.])
        coeff=dict(trusted_acc=base,**{m:dict(position=[t,0.],velocity=[1.,0.],acceleration=[500.,500.]) for m in ADDED})
        q=dict(t=t,valid=True,coefficients=coeff,predictions={m:{str(h):prior.evaluate(c,h) for h in [.1,.2,.4]} for m,c in coeff.items()},
            event_force_diagnostic=dict(models={m:dict(active=True,updates=30) for m in ADDED}))
        out=g.step(q,dict(detected=True,confidence=.9,center=[t,0.]))
        for m in ADDED70:assert out['coefficients'][m]==base
    assert all(len(g.errors[h])>=5 for h in [.2,.4])
    return dict(worse_physics_rejected=True,history_exact_when_rejected=True,
                delayed_feedback_matured=True,future_truth_imported=False)
