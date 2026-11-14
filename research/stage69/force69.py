"""2D effective force and online residual correction from center-free activity.

Adds branches after the unchanged62 predictor/selector. All variants use the
same observed positions, direction, activity admission and history fallback.
The matched constant-input controls isolate the extra event input.
"""
from collections import deque
from pathlib import Path
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage62'),str(ROOT/'research/stage68'),str(ROOT/'research/stage6')]
from independent62 import Managed62,Selector,MODELS as ORIGINAL_MODELS,prior
from feedback52 import DelayedFeedback
from activity68 import Reference
from silhouette import angle_distance
from event_drone_forecast.paper_config import PaperConfig

KINDS=['density','periodic']
ADDED=[f'{family}_{kind}{suffix}' for family in ['force','residual'] for kind in KINDS for suffix in ['_constant','']]
MODELS=ORIGINAL_MODELS+ADDED
PARAMS=dict(axis_expiry_s=.16,body_confirmation=3,body_step_gap_s=.04,body_change_deg=12.,
    body_quality_min=.45,feedback_delay_s=.2,feedback_expiry_s=.26,warmup_s=.6,
    feedback_confidence_min=.5,rls_scale=200.,residual_initial_covariance=.5,
    residual_forgetting=.999,residual_error_cap=2.,effective_gain_bounds=[0.,5000.],
    residual_acceleration_cap=2500.,history_fallback='trusted_acc',
    scope='2D projected force direction; upright/temporal sign assumption. '
          'No recovered3D pose or metric wind, mass, RPM, thrust coefficient. '
          'Density squared and weighted local f squared are experimental inputs. '
          'Absolute force reuses52 RLS; residual starts at pure history and learns '
          'effective force plus2D bias from delayed current detector positions.')


class Direction:
    def __init__(self):self.history=deque(maxlen=3);self.normal=None;self.t=None;self.quality=0.;self.source=None
    def step(self,t,q,body,rotor_quality):
        current=False;proposed=None;quality=0.;source=None
        if body.get('valid') and body.get('confidence',0)>=.45:
            a=body['angle_rad']
            if self.history and (t-self.history[-1][0]>.04000001 or np.degrees(angle_distance(a,self.history[-1][1]))>12):self.history.clear()
            self.history.append((t,a))
            if len(self.history)==3:
                a=float(np.angle(np.mean([np.exp(2j*a) for _,a in self.history]))/2)
                proposed=np.array([-np.sin(a),np.cos(a)]);quality=body['confidence'];source='confirmed_body_line'
        else:self.history.clear()
        if proposed is None:
            d=q['feature_diagnostic'];m=d['models']['adaptive_axis_only']
            if d['axis_age_s'] is not None and abs(d['axis_age_s'])<1e-8 and m['normal'] is not None:
                v=np.asarray(m['normal'][:2],float);length=np.linalg.norm(v)
                if length>.1:
                    proposed=v/length;quality=float(rotor_quality);source='current_rotor_projection'
        if proposed is not None:
            if self.normal is None:
                if proposed[1]>0:proposed=-proposed
            elif proposed@self.normal<0:proposed=-proposed
            self.normal=proposed;self.t=t;self.quality=float(quality);self.source=source;current=True
        active=self.t is not None and t-self.t<=.16000001
        return (self.normal.copy() if active else None),dict(active=active,current=current,quality=self.quality,
            age_s=None if self.t is None else t-self.t,source=self.source,
            interpretation='Unit projected2D force feature, not a3D plane-normal estimate.')


class Residual:
    def __init__(self):
        self.theta=np.zeros(3);self.cov=np.eye(3)*.5;self.queue=deque();self.started=None;self.updates=0
    @staticmethod
    def matrix(n,s):return np.array([[s*n[0],1.,0.],[s*n[1],0.,1.]])
    def step(self,t,base,n,s,launch,feedback,center):
        if self.started is None:self.started=t
        while self.queue and t-self.queue[0][0]>.26000001:self.queue.popleft()
        if feedback:
            while self.queue and t-self.queue[0][0]>=.2-1e-8:
                ot,old_base,old_n,old_s=self.queue.popleft();dt=t-ot
                target=2*(np.asarray(center)-np.asarray(prior.evaluate(old_base,dt)))/dt**2
                X=self.matrix(old_n,old_s)
                error=np.clip(target/200.-X@self.theta,-2.,2.)
                gain=np.linalg.solve(np.eye(2)*.5+X@self.cov@X.T,X@self.cov).T
                self.theta+=gain@error
                self.theta[0]=np.clip(self.theta[0],0.,25.)
                self.theta[1:]=np.clip(self.theta[1:],-12.5,12.5)
                self.cov=(self.cov-gain@X@self.cov)/.999
                self.cov=(self.cov+self.cov.T)/2
                values,vectors=np.linalg.eigh(self.cov);self.cov=vectors@np.diag(np.clip(values,1e-6,100))@vectors.T
                self.updates+=1
        if launch and t-self.started>=.6:self.queue.append((t,copy.deepcopy(base),n.copy(),s))
    def acceleration(self,n,s):return np.clip(200.*self.matrix(n,s)@self.theta,-2500.,2500.)


class ForceForecast:
    def __init__(self):self.reset()
    def reset(self):
        self.direction=Direction();self.references={k:Reference() for k in KINDS};self.last_seen=None
        self.models={m:(DelayedFeedback(PaperConfig(),True) if m.startswith('force_') else Residual()) for m in ADDED}
    def step(self,original,obs,body,activity,rotor_quality):
        t=original['t'];q=copy.deepcopy(original)
        if self.last_seen is not None and t-self.last_seen>.5:self.reset()
        observed=bool(obs.get('detected'))
        if observed:self.last_seen=t
        n,nd=self.direction.step(t,original,body,rotor_quality)
        # Reference never consumes future samples and remains fixed after5.
        density=activity.get('density')
        values=dict(density=density*density if density is not None and density>0 else None,
                    periodic=activity.get('frequency_squared'))
        features={k:self.references[k].step(t,v) for k,v in values.items()}
        diagnostic=dict(direction=nd,features=features,models={})
        for name,model in self.models.items():
            kind='density' if '_density' in name else 'periodic';feature=features[kind]
            s=1. if name.endswith('_constant') else feature['ratio']
            active=bool(q['valid'] and n is not None and feature['active'])
            feedback=bool(observed and obs.get('confidence',0)>.5 and q['valid'])
            launch=bool(active and feedback and nd['current'] and feature['current'] and nd['quality']>.45)
            if q['valid']:
                base=q['coefficients']['trusted_acc'];cv=q['coefficients']['cv']
                if name.startswith('force_'):
                    state=np.r_[obs['center'] if observed else cv['position'],cv['velocity']]
                    model.update(t,state,np.r_[n,0.] if n is not None else None,s if s is not None else 1.,
                                 nd['quality'],feedback,launch)
                    if active:
                        times,points=model.rollout(cv['position'],cv['velocity'],np.r_[n,0.],np.zeros(3),s)
                        c=dict(copy.deepcopy(cv),acceleration=model.acceleration(n,s,np.asarray(cv['velocity'])).tolist(),
                               rollout_times=times.tolist(),rollout_positions=points.tolist())
                    else:c=copy.deepcopy(base)
                    parameters=model.parameters.tolist();bias=model.disturbance.tolist()
                else:
                    model.step(t,base,n,s,launch,feedback,obs.get('center'))
                    c=copy.deepcopy(base)
                    if active:
                        a=model.acceleration(n,s);c['acceleration']=(np.asarray(c['acceleration'])+a).tolist()
                    parameters=(model.theta*200).tolist();bias=None
                q['coefficients'][name]=c
            else:parameters=None;bias=None
            diagnostic['models'][name]=dict(active=active,launch=launch,updates=model.updates,
                ratio=s,effective_parameters=parameters,disturbance=bias,queue=len(model.queue))
        if q['valid']:
            q['predictions'].update({m:{str(h):prior.evaluate(q['coefficients'][m],h) for h in prior.PARAMS['horizons']} for m in ADDED})
        q['event_force_diagnostic']=diagnostic
        return q


def controls():
    # Known delayed acceleration driven by an exogenous force proxy; never
    # give the estimator the true parameters. This checks wiring, not reality.
    results=[]
    for signal in ['varying','constant']:
        model=Residual();p=np.zeros(2);v=np.zeros(2);dt=.02;n=np.array([.6,-.8]);history=[]
        for j in range(350):
            t=j*dt;s=1.+.45*np.sin(t*3.) if signal=='varying' else 1.
            base=dict(position=p.tolist(),velocity=v.tolist(),acceleration=[0.,0.])
            model.step(t,base,n,s,True,True,p)
            a=300*s*n+np.array([20.,140.]);p=p+v*dt+.5*a*dt**2;v+=a*dt
            history.append(model.updates)
        assert model.updates>0 and np.all(np.isfinite(model.theta))
        results.append(dict(signal=signal,updates=model.updates,parameters=(model.theta*200).tolist()))
    # No current position => no regression or learning; missing axis => no rollouts.
    model=Residual()
    for j in range(100):model.step(j*.02,dict(position=[0,0],velocity=[0,0],acceleration=[0,0]),None,None,False,False,None)
    assert model.updates==0 and len(model.queue)==0
    return dict(known_force_wiring=results,no_feedback_no_learning=True,
                caveat='Integration/unit control only; final RAW forecast scores decide efficacy.')
