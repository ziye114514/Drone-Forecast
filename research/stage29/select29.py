"""Delayed exponential expert weighting, reusing local paper_bank constants."""
from collections import deque
from pathlib import Path
import copy,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage27'))
import physics27 as prior

HISTORY=['cv','bounded_acc','trusted_acc']
EXPERTS=HISTORY+['adaptive_axis_only','adaptive_axis_freq']
ADDED=['fixed_history_mix','adaptive_history_mix','fixed_all_mix','adaptive_all_mix','adaptive_pick']
MODELS=prior.MODELS+ADDED
PARAMS=dict(delay_s=.2,maximum_feedback_age_s=.26,learning_rate=.08,loss_scale_px=35.,loss_clip=16.,fixed_share=.01,
    observation_confidence_min=.5,cv_initial_weight=.9,history=HISTORY,experts=EXPERTS,horizons=prior.PARAMS['horizons'],
    origin='Existing src/event_drone_forecast/paper_bank.py delayed weights (.2s,.26s,35px,.08,clip16,share.01). CV prior .9, other initial mass equally divided. No tuning on stage29 scores.')


def initial(names):return np.array([PARAMS['cv_initial_weight']]+[(1-PARAMS['cv_initial_weight'])/(len(names)-1)]*(len(names)-1))


def blend(coefficients,names,weights):
    times=np.linspace(0,.5,26)
    points=sum(w*np.array([prior.evaluate(coefficients[m],float(t)) for t in times]) for m,w in zip(names,weights))
    cv=coefficients['cv']
    return dict(position=copy.deepcopy(cv['position']),velocity=copy.deepcopy(cv['velocity']),
        acceleration=sum(w*np.asarray(coefficients[m]['acceleration']) for m,w in zip(names,weights)).tolist(),
        rollout_times=times.tolist(),rollout_positions=points.tolist())


class Selector:
    def __init__(self):self.last_seen=None;self.reset()

    def reset(self):
        self.weights={'history':initial(HISTORY),'all':initial(EXPERTS)};self.queue=deque();self.updates=0;self.last_seen=None

    def step(self,forecast,observation):
        t=forecast['t'];observed=bool(observation.get('detected'))
        if self.last_seen is not None and t-self.last_seen>prior.robust.PARAMS['reset_gap_s']:self.reset()
        if observed:self.last_seen=t
        while self.queue and t-self.queue[0][0]>PARAMS['maximum_feedback_age_s']:self.queue.popleft()
        feedback=False;loss_record=None
        if observed and observation.get('confidence',0)>PARAMS['observation_confidence_min']:
            while self.queue and t-self.queue[0][0]>=PARAMS['delay_s']-1e-8:
                old_t,old=self.queue.popleft();elapsed=t-old_t;z=np.asarray(observation['center'])
                loss={m:float(np.clip(np.sum((np.asarray(prior.evaluate(old[m],elapsed))-z)**2)/PARAMS['loss_scale_px']**2,0,PARAMS['loss_clip'])) for m in EXPERTS}
                for group,names in [('history',HISTORY),('all',EXPERTS)]:
                    weights=self.weights[group]*np.exp(-PARAMS['learning_rate']*np.array([loss[m] for m in names]));weights/=weights.sum()
                    self.weights[group]=(1-PARAMS['fixed_share'])*weights+PARAMS['fixed_share']/len(names)
                self.updates+=1;feedback=True;loss_record=dict(anchor_t=old_t,elapsed_s=elapsed,loss=loss)
        q=copy.deepcopy(forecast)
        if q['valid']:
            c=q['coefficients']
            variants={'fixed_history_mix':blend(c,HISTORY,initial(HISTORY)),
                'adaptive_history_mix':blend(c,HISTORY,self.weights['history']),
                'fixed_all_mix':blend(c,EXPERTS,initial(EXPERTS)),
                'adaptive_all_mix':blend(c,EXPERTS,self.weights['all'])}
            chosen=EXPERTS[int(np.argmax(self.weights['all']))]
            variants['adaptive_pick']=copy.deepcopy(c[chosen]);q['coefficients'].update(variants)
            self.queue.append((t,{m:copy.deepcopy(c[m]) for m in EXPERTS}))
            q['predictions']={m:{str(h):prior.evaluate(v,h) for h in PARAMS['horizons']} for m,v in c.items()}
        else:chosen=None
        q['selection_diagnostic']=dict(updates=self.updates,feedback_used=feedback,last_feedback=loss_record,
            weights={'history':dict(zip(HISTORY,self.weights['history'].tolist())),'all':dict(zip(EXPERTS,self.weights['all'].tolist()))},
            chosen_expert=chosen,feedback_reference='Current detector center after past forecasts mature, not independent GT.')
        return q
