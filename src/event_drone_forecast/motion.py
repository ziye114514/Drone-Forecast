"""Causal state filtering and delayed online projection-dynamics identification."""
from collections import deque
import numpy as np

BASE_METHODS=('cv','ca','adaptive_history','adaptive_rotor')
METHODS=BASE_METHODS+('adaptive','paper_nominal','paper_adaptive','paper_fused')

class KinematicFilter:
    def __init__(self,cfg,order=2):
        self.cfg,self.order=cfg,order
        self.x=None;self.P=None;self.t=None

    def update(self,t,z=None,confidence=1.,measurement_covariance=None):
        n=self.order*2
        if self.x is None:
            if z is None:return None
            self.x=np.zeros(n);self.x[:2]=z
            self.P=np.diag([self.cfg.position_sigma**2]*2+[1e6]*(n-2));self.t=t
            return self.x.copy()
        dt=t-self.t
        if dt<0:raise ValueError('Time must be monotonic')
        F=np.eye(n);F[0:2,2:4]=np.eye(2)*dt
        if self.order==3:
            F[0:2,4:6]=np.eye(2)*dt*dt/2;F[2:4,4:6]=np.eye(2)*dt
            b=np.array([dt**3/6,dt*dt/2,dt]); sigma=self.cfg.acceleration_sigma*5
        else:
            b=np.array([dt*dt/2,dt]);sigma=self.cfg.acceleration_sigma
        Q=np.kron(np.outer(b,b)*sigma*sigma,np.eye(2))
        self.x=F@self.x;self.P=F@self.P@F.T+Q;self.t=t
        if z is not None:
            H=np.zeros((2,n));H[:,:2]=np.eye(2)
            R=np.eye(2)*self.cfg.position_sigma**2/max(confidence,.25)**2
            if measurement_covariance is not None:
                extra=np.asarray(measurement_covariance,float)
                if extra.shape==(2,2) and np.isfinite(extra).all():
                    values,vectors=np.linalg.eigh((extra+extra.T)/2)
                    R+=vectors@np.diag(np.clip(values,0,2500))@vectors.T
            K=np.linalg.solve(H@self.P@H.T+R,H@self.P).T
            innovation=np.asarray(z)-H@self.x
            self.x+=K@innovation
            A=np.eye(n)-K@H;self.P=A@self.P@A.T+K@R@K.T
        return self.x.copy()

class RecursiveDynamics:
    def __init__(self,cfg,with_rotors=False):
        self.cfg,self.with_rotors=cfg,with_rotors
        self.n=11 if with_rotors else 5
        self.weights=np.zeros((self.n,2))
        self.weights[3,0]=.5;self.weights[4,1]=.5  # prior = CA acceleration / 2000
        self.cov=np.eye(self.n)*1.5
        self.disturbance=np.zeros(2)
        self.queue=deque();self.updates=0;self.last_update=None
        self.started=None
        self.update_allowed=False
        self.frozen_frames=0

    def features(self,cv,ca,obs):
        a=np.clip(ca[4:6],-6000,6000)
        x=[1.,cv[2]/600,cv[3]/600,a[0]/1000,a[1]/1000]
        if self.with_rotors:
            n=np.array(obs.get('normal_xy',[0,0]),float)
            # A circle/line yields an undirected axis. Doubled-angle features
            # remain identical under n -> -n; no arbitrary arrow is physical.
            n=np.array([n[0]**2-n[1]**2,2*n[0]*n[1]])
            s=float(obs.get('frequency_squared_relative',1))
            quality=float(obs.get('frequency_quality',0))*float(obs.get('geometry_quality',1.))*float(obs.get('confidence',0))
            x.extend([n[0]*s*quality,n[1]*s*quality,(s-1)*quality,
                      n[0]*quality,n[1]*quality,quality])
        return np.clip(np.asarray(x),-10,10)

    def update(self,t,cv,ca,obs):
        phi=self.features(cv,ca,obs)
        if self.started is None:self.started=t
        observed=obs.get('detected') and obs.get('confidence',0)>.35
        geometry_valid=(obs.get('geometry_quality',1.)>=self.cfg.geometry_learning_min_quality
                        and obs.get('frequency_quality',0)>=self.cfg.frequency_learning_min_quality
                        and obs.get('center_sigma_px',0)<25.)
        self.update_allowed=bool(observed and (geometry_valid or not self.with_rotors))
        if self.with_rotors and not self.update_allowed:self.frozen_frames+=1
        while self.queue and t-self.queue[0][0]>self.cfg.learning_delay_s+.08:self.queue.popleft()
        if self.update_allowed and t-self.started>.4:
            while self.queue and t-self.queue[0][0]>=self.cfg.learning_delay_s:
                old_t,p,v,x,confidence=self.queue.popleft()
                dt=t-old_t
                if dt>self.cfg.learning_delay_s+.08:continue
                # New observation corrects only subsequent predictions, never an old forecast.
                target=2*(cv[:2]-p-v*dt)/(dt*dt)
                target=np.clip(target,-self.cfg.max_acceleration_px_s2,self.cfg.max_acceleration_px_s2)/2000
                estimate=x@self.weights
                error=np.clip(target-estimate,-2.,2.)
                Px=self.cov@x
                gain=Px/(self.cfg.forgetting_factor+x@Px+1./max(confidence,.1))
                self.weights+=np.outer(gain,error)
                self.cov=(self.cov-np.outer(gain,x@self.cov))/self.cfg.forgetting_factor
                self.cov=(self.cov+self.cov.T)/2
                elapsed=(t-self.last_update) if self.last_update is not None else .02
                beta=1-np.exp(-elapsed/self.cfg.disturbance_time_s)
                residual=(target-x@self.weights)*2000
                self.disturbance=(1-beta)*self.disturbance+beta*residual
                self.last_update=t;self.updates+=1
            self.queue.append((t,cv[:2].copy(),cv[2:4].copy(),phi.copy(),obs['confidence']))
        return np.clip(phi@self.weights*2000+self.disturbance,
                       -self.cfg.max_acceleration_px_s2,self.cfg.max_acceleration_px_s2)

class ForecastBank:
    def __init__(self,cfg):
        self.cfg=cfg;self.cv=KinematicFilter(cfg,2);self.ca=KinematicFilter(cfg,3)
        self.learners={m:RecursiveDynamics(cfg,m=='adaptive_rotor') for m in BASE_METHODS[2:]}
        self.last_seen=None;self.last_obs={};self.track_id=0
        self.expert_weights=np.array([.55,.2,.1,.15])
        self.expert_queue=deque()

    def reset(self):
        track_id=self.track_id+1
        learned={name:(q.weights.copy(),q.cov.copy()) for name,q in self.learners.items()}
        expert_weights=self.expert_weights.copy()
        self.__init__(self.cfg);self.track_id=track_id
        # Single-target sequence assumption: retain learned projection parameters across gaps,
        # discard stale states, delayed samples, and transient disturbance estimates.
        for name,(weights,cov) in learned.items():
            self.learners[name].weights=weights;self.learners[name].cov=cov
        self.expert_weights=expert_weights

    def step(self,obs):
        t=obs['t'];detected=obs.get('detected',False)
        if self.last_seen is not None and t-self.last_seen>self.cfg.reset_gap_s:
            self.reset()
        if detected:
            self.last_seen=t;self.last_obs=obs
        if self.last_seen is None:return None
        z=obs['center'] if detected else None
        covariance=obs.get('center_covariance_px2') if detected else None
        cv=self.cv.update(t,z,obs.get('confidence',0),covariance);ca=self.ca.update(t,z,obs.get('confidence',0),covariance)
        if cv is None:return None
        active=t-self.last_seen<=self.cfg.max_coast_s
        accels={'cv':np.zeros(2),'ca':np.clip(ca[4:6],-self.cfg.max_acceleration_px_s2,self.cfg.max_acceleration_px_s2)}
        current={**self.last_obs,**obs}
        for name,learner in self.learners.items():
            accels[name]=learner.update(t,cv,ca,current)
        if not self.learners['adaptive_rotor'].update_allowed:
            # An unreliable geometry measurement must not drive a fresh turn.
            accels['adaptive_rotor']=accels['adaptive_history'].copy()
        # All methods share the same detector. CA uses its own causal filter state.
        states={name:dict(position=(ca[:2] if name=='ca' else cv[:2]).tolist(),
                velocity=(ca[2:4] if name=='ca' else cv[2:4]).tolist(),acceleration=accels[name].tolist()) for name in BASE_METHODS}
        if detected and obs.get('confidence',0)>.35:
            while self.expert_queue and t-self.expert_queue[0][0]>=.2:
                previous_t,previous_states=self.expert_queue.popleft()
                elapsed=t-previous_t
                if elapsed>.28:continue
                # Completed past forecasts are scored against new event-derived measurements.
                losses=np.array([np.linalg.norm(extrapolate(previous_states[m],elapsed)-cv[:2])**2 for m in BASE_METHODS])
                losses=np.clip(losses/(35.**2),0,16)
                self.expert_weights*=np.exp(-.08*losses)
                self.expert_weights/=max(self.expert_weights.sum(),1e-12)
                self.expert_weights=.98*self.expert_weights+.02/len(BASE_METHODS)
            self.expert_queue.append((t,states.copy()))
        states['adaptive']={key:sum(self.expert_weights[i]*np.array(states[name][key]) for i,name in enumerate(BASE_METHODS)).tolist() for key in ('position','velocity','acceleration')}
        spread=np.mean([np.sum((accels[m]-np.array(states['adaptive']['acceleration']))**2) for m in BASE_METHODS])
        intervals={}
        for horizon in self.cfg.horizons:
            H=np.column_stack((np.eye(2),np.eye(2)*horizon))
            covariance=H@self.cv.P@H.T+np.eye(2)*spread*horizon**4/4
            intervals[str(horizon)]=float(2.45*np.sqrt(max(np.linalg.eigvalsh(covariance)[-1],0)))
        return dict(t=t,active=active,observed=detected,track_id=self.track_id,states=states,
                    uncertainty_px=float(np.sqrt(max(np.trace(self.cv.P[:2,:2])/2,0))),
                    online_updates={k:v.updates for k,v in self.learners.items()},
                    disturbance_px_s2=self.learners['adaptive_rotor'].disturbance.tolist(),
                    projected_rotor_weights=self.learners['adaptive_rotor'].weights[5:].tolist(),
                    rotor_learning_allowed=self.learners['adaptive_rotor'].update_allowed,
                    rotor_learning_frozen_frames=self.learners['adaptive_rotor'].frozen_frames,
                    forecast_uncertainty_radius_px=intervals,
                    uncertainty_interpretation='Uncalibrated covariance/model-spread diagnostic; not verified 95% coverage.',
                    expert_weights=dict(zip(BASE_METHODS,self.expert_weights.tolist())))

def extrapolate(state,horizon):
    if 'rollout_times' in state:
        times=np.asarray(state['rollout_times']);points=np.asarray(state['rollout_positions'])
        if 0<=horizon<=times[-1]+1e-8:
            return np.array([np.interp(horizon,times,points[:,i]) for i in range(2)])
    return np.asarray(state['position'])+np.asarray(state['velocity'])*horizon+.5*np.asarray(state['acceleration'])*horizon*horizon
