"""Causal horizon-specific image-displacement calibration, not thrust fitting.

Only predictions already due can learn from current strong detector boxes.
Native-time interpolation uses two measurements already available. The bounded
linear solve is SciPy's existing implementation; no manual labels enter here.
"""
from collections import deque
from threading import RLock
import copy
import numpy as np
from scipy.optimize import lsq_linear
from live_motion import predictions
from event_drone_forecast.position.rollout import forecast as base_forecast

PARAMS=dict(horizons=(.1,.2,.4),gain_bounds=(0.,1.5),minimum_updates=5,
            maximum_feedback_gap_s=.08,minimum_displacement_px=6.,
            minimum_relative_past_loss_gain=.05,reset_gap_s=.5,
            prior_gain=1.,prior_weight=1.)


class HorizonGain:
    def __init__(self,publish_hz=5.):
        self.publish_hz=publish_hz;self.window=max(1.,5./publish_hz)
        self.lock=RLock();self._clear()

    def _clear(self):
        self.pending=[];self.previous=None;self.last_available=None
        self.samples={h:deque() for h in PARAMS['horizons']}
        self.losses={h:deque() for h in PARAMS['horizons']}
        self.versions=deque(maxlen=64);self.updates={h:0 for h in PARAMS['horizons']}

    def observe(self,t,point,available_t,strong=True):
        if not strong or point is None:return
        if available_t<t-1e-8:raise ValueError('Feedback available before observation')
        with self.lock:
            if self.previous is not None and t-self.previous[0]>PARAMS['reset_gap_s']:self._clear()
            if self.previous is not None and t<=self.previous[0]:raise ValueError('Increasing feedback times required')
            point=np.asarray(point,float);remaining=[]
            for old in self.pending:
                if old['target_t']>t+1e-8:remaining.append(old);continue
                target=None
                if abs(old['target_t']-t)<1e-8:target=point
                elif self.previous is not None:
                    a,p=self.previous
                    if a<old['target_t']<=t and t-a<=PARAMS['maximum_feedback_gap_s']+1e-8:
                        alpha=(old['target_t']-a)/(t-a);target=p+alpha*(point-p)
                if target is None:continue
                h=old['h'];d=old['delta'];norm=float(np.linalg.norm(d))
                if norm<PARAMS['minimum_displacement_px']:continue
                self.samples[h].append((old['target_t'],d/norm,(target-old['anchor'])/norm))
                lb=float(np.sum((old['baseline']-target)**2));lc=float(np.sum((old['proposed']-target)**2))
                self.losses[h].append((old['target_t'],lb,lc));self.updates[h]+=1
            self.pending=remaining;self.previous=(t,point);self.last_available=available_t
            params={}
            for h in PARAMS['horizons']:
                for queue in (self.samples[h],self.losses[h]):
                    while queue and t-queue[0][0]>self.window+1e-8:queue.popleft()
                gain=1.;n=len(self.samples[h]);active=False;old_loss=new_loss=None
                if n>=PARAMS['minimum_updates']:
                    A=np.r_[np.concatenate([s[1] for s in self.samples[h]]),[1.]].reshape(-1,1)
                    b=np.r_[np.concatenate([s[2] for s in self.samples[h]]),[1.]]
                    fit=lsq_linear(A,b,bounds=PARAMS['gain_bounds'],method='bvls')
                    if fit.success and np.isfinite(fit.x).all():gain=float(fit.x[0])
                    old_loss=float(np.mean([s[1] for s in self.losses[h]]));new_loss=float(np.mean([s[2] for s in self.losses[h]]))
                    active=bool(new_loss<(1.-PARAMS['minimum_relative_past_loss_gain'])*old_loss)
                params[str(h)]=dict(gain=gain,active=active,samples=n,total_updates=self.updates[h],
                                    past_baseline_MSE=old_loss,past_proposed_MSE=new_loss)
            self.versions.append((available_t,t,params))

    def forecast(self,publication_t,c,age,valid=True,enqueue=True):
        baseline,_=predictions(c,age)
        if c is None:return baseline,dict(active=False,parameters={})
        with self.lock:
            eligible=[version for version in self.versions if version[0]<=publication_t+1e-9]
            version=eligible[-1] if eligible else None
            params=copy.deepcopy(version[2]) if version else {str(h):dict(gain=1.,active=False,samples=0,total_updates=0) for h in PARAMS['horizons']}
            anchor=np.asarray(base_forecast(c,age),float);answer={}
            for h in PARAMS['horizons']:
                base=np.asarray(baseline[str(h)],float);d=base-anchor;p=params[str(h)]
                proposed=anchor+p['gain']*d
                answer[str(h)]=(proposed if p['active'] else base).tolist()
                if valid and enqueue:
                    self.pending.append(dict(origin_t=publication_t,target_t=publication_t+h,h=h,
                                             anchor=anchor,delta=d,baseline=base,proposed=proposed))
            return answer,dict(parameters=params,feedback_available_t=version[0] if version else None,
                               last_feedback_observation_t=version[1] if version else None,
                               active=any(p['active'] for p in params.values()),force_parameters_identified=False)
