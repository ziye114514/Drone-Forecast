"""Observation-error interventions, without modifying the frozen v0.3 runtime.

All predictors share the current position/velocity estimate. This isolates
learning/forcing effects from differences between position filters. The oracle
state and oracle learner reset are explicitly diagnostic, not deployable.
"""
from pathlib import Path
from collections import deque
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.config import Config
from event_drone_forecast.motion import KinematicFilter
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.paper_physics import ProjectedThrustModel,weak_normal

METHODS=('cv','fixed','ordinary_rls','quality_rls','v030_rls','ordinary_reset_oracle')
SCENARIOS=('clean','double_flagged','double_confident','half_confident','ellipse_flagged',
           'ellipse_confident','missing_two','id_permutation_only','center_glitch_confident')
HORIZONS=np.array([.1,.2,.4])
FAULT_START=5.;FAULT_END=6.
SCALES=np.array([220.,200.,1.,1.])

def exact_rollout(position,velocity,normal,ratio,model,horizons=HORIZONS):
    """Constant current normal/RPM, exact linear drag. No future observations."""
    p=np.asarray(position);v=np.asarray(velocity);n=np.asarray(normal)
    k,g,dx,dy=model.parameters
    force=k*ratio*n[:2]-g*np.asarray(model.cfg.image_up)
    if model.adaptive:force+=model.disturbance
    force=np.clip(force,-model.cfg.max_acceleration_px_s2,model.cfg.max_acceleration_px_s2)
    h=np.asarray(horizons);result=np.empty((len(h),2))
    for j,d in enumerate((dx,dy)):
        if d<1e-8:result[:,j]=p[j]+v[j]*h+.5*force[j]*h*h
        else:
            gain=-np.expm1(-d*h)/d
            result[:,j]=p[j]+gain*v[j]+(h-gain)/d*force[j]
    return result

class DelayedRLS(ProjectedThrustModel):
    """Controlled benchmark RLS, not a new proposed research method.

    ordinary and quality versions have identical equations, priors, constraints,
    residual smoothing and forgetting. Only admission of low-quality/missing-
    rotor samples differs. The legacy model is tested separately unchanged.
    """
    def __init__(self,cfg,gated=False):
        super().__init__(cfg,True);self.gated=gated

    def update(self,t,state,n,s,quality,observed,four_visible):
        if self.started is None:self.started=t
        if self.reference_normal is None and n is not None and four_visible and quality>.35:
            self.reference_normal=np.asarray(n).copy()
            self.parameters[0]=self.parameters[1]/max(.2,float(np.asarray(n[:2])@np.asarray(self.cfg.image_up)))
        self.update_allowed=bool(observed and n is not None and (not self.gated or (four_visible and quality>.45)))
        while self.queue and t-self.queue[0][0]>self.cfg.adaptation_delay_s+.06:self.queue.popleft()
        if not self.update_allowed or t-self.started<.6:return
        while self.queue and t-self.queue[0][0]>=self.cfg.adaptation_delay_s-1e-9:
            old_t,p,v,old_n,old_s,old_q=self.queue.popleft();dt=t-old_t
            target=np.clip(2*(state[:2]-p-v*dt)/dt**2,-1500,1500)
            up=np.asarray(self.cfg.image_up)
            matrix=np.array([[old_s*old_n[0],-up[0],-v[0],0.],
                             [old_s*old_n[1],-up[1],0.,-v[1]]])
            scale=np.array([200.,200.,1.,1.]);theta=self.parameters/scale;X=matrix*scale/200.
            error=np.clip((target-matrix@self.parameters)/200.,-2,2)
            # Ordinary and quality arms use the SAME fixed measurement noise.
            gain=np.linalg.solve(np.eye(2)*.5+X@self.cov@X.T,X@self.cov).T
            theta+=gain@error;self.parameters=theta*scale
            self.parameters[:2]=np.clip(self.parameters[:2],20,5000);self.parameters[2:]=np.clip(self.parameters[2:],0,3)
            self.cov=(self.cov-gain@X@self.cov)/self.cfg.adaptation_forgetting
            self.cov=(self.cov+self.cov.T)/2
            values,vectors=np.linalg.eigh(self.cov);self.cov=vectors@np.diag(np.clip(values,1e-6,100))@vectors.T
            residual=target-self.acceleration(old_n,old_s,v,False)
            self.disturbance=.97*self.disturbance+.03*np.clip(residual,-300,300)
            self.last_update=t;self.updates+=1
        self.queue.append((t,state[:2].copy(),state[2:4].copy(),np.asarray(n).copy(),s,quality))

def make_world(seed,world='matched',duration=14.,dt=.02):
    rng=np.random.default_rng(seed);times=np.arange(round(duration/dt)+1)*dt;N=len(times)
    angle=.20*np.sin(.8*times)+.07*np.sin(1.9*times)
    aspect=.4+.025*np.sin(.6*times)
    normals=np.column_stack((np.sin(angle)*np.sqrt(1-aspect**2),-np.cos(angle)*np.sqrt(1-aspect**2),aspect))
    relative=1.+.12*np.sin(.9*times)+.04*np.sin(1.7*times)
    rotor_f0=300*np.array([1.015,.985,1.01,.99]);frequencies=np.sqrt(relative[:,None])*rotor_f0
    k=200/np.sqrt(1-.4**2)*(1.3 if world=='gain_mismatch' else 1.)
    acceleration=k*relative[:,None]*normals[:,:2]-200*np.array([0.,-1.])
    position=np.empty((N,2));velocity=np.empty_like(position);position[0]=[400,300];velocity[0]=[35,-5]
    for i in range(1,N):
        position[i]=position[i-1]+velocity[i-1]*dt+.5*acceleration[i-1]*dt**2
        velocity[i]=velocity[i-1]+acceleration[i-1]*dt
    return dict(seed=seed,world=world,times=times,position=position,velocity=velocity,acceleration=acceleration,
        angle=angle,aspect=aspect,normal=normals,relative=relative,frequencies=frequencies,
        noisy_position=position+rng.normal(0,.75,(N,2)),
        noisy_frequency=frequencies*(1+rng.normal(0,.006,(N,4))),
        noisy_angle=angle+rng.normal(0,np.deg2rad(.5),N),
        noisy_aspect=np.clip(aspect+rng.normal(0,.008,N),.1,.9),true_parameters=np.array([k,200,0,0]))

def sensor_sample(world,i,scenario='clean',start=FAULT_START,end=FAULT_END):
    t=world['times'][i];fault=bool(start<=t<end);f=world['noisy_frequency'][i].copy()
    angle=float(world['noisy_angle'][i]);aspect=float(world['noisy_aspect'][i]);quality=.95;four=True;ids=np.arange(4)
    z=world['noisy_position'][i].copy()
    if fault:
        if scenario.startswith('double'):f*=2
        if scenario=='half_confident':f*=.5
        if scenario.startswith('ellipse'):angle+=np.deg2rad(35);aspect=.15
        if scenario.endswith('flagged'):quality=.1
        if scenario=='missing_two':four=False
        if scenario=='id_permutation_only':ids=np.array([2,3,0,1])
        if scenario=='center_glitch_confident':z+=np.array([15.,-10.])
    rotor=dict(angle=float(np.rad2deg(angle)),ratio=aspect)
    return dict(t=float(t),position=z,frequencies=f,normal=weak_normal(rotor),quality=quality,
                observed=True,four_visible=four,four_frequencies=four,ids=ids,fault=fault)

class ComparisonBank:
    def __init__(self,dt=.02):
        self.cfg=PaperConfig(step_us=round(dt*1e6));self.cv=KinematicFilter(Config(step_us=round(dt*1e6)),2)
        self.models={'fixed':ProjectedThrustModel(self.cfg,False),
            'ordinary_rls':DelayedRLS(self.cfg,False),'quality_rls':DelayedRLS(self.cfg,True),
            'v030_rls':ProjectedThrustModel(self.cfg,True),'ordinary_reset_oracle':DelayedRLS(self.cfg,False)}
        self.reference=None;self.last_s=1.;self.last_force_t=None;self.latest_state=None
        self.last_n=np.array([0.,-np.sqrt(.84),.4])

    def step(self,sample,oracle_state=None,reset_model=None):
        t=sample['t'];z=sample['position'] if sample['observed'] else None
        state=self.cv.update(t,z,sample.get('position_confidence',.95),sample.get('position_covariance'))
        if state is None:return None
        if oracle_state is not None:state=np.asarray(oracle_state).copy()
        self.latest_state=state.copy();n=np.asarray(sample['normal']);self.last_n=n.copy()
        if sample['four_frequencies']:
            total=float(np.sum(np.asarray(sample['frequencies'])**2))
            if self.reference is None:self.reference=total
            self.last_s=float(np.clip(total/self.reference,.2,5));self.last_force_t=t
        fresh=self.last_force_t is not None and t-self.last_force_t<=.16+1e-9
        active=bool(fresh and sample['quality']>.15)
        if reset_model is not None:self.models['ordinary_reset_oracle']=copy.deepcopy(reset_model)
        prediction=np.empty((len(METHODS),len(HORIZONS),2));parameters=np.zeros((len(METHODS),4));disturbance=np.zeros((len(METHODS),2));allowed=np.zeros(len(METHODS),bool);updates=np.zeros(len(METHODS),int)
        prediction[0]=state[:2]+HORIZONS[:,None]*state[2:]
        for j,name in enumerate(METHODS[1:],1):
            model=self.models[name];model.update(t,state,n,self.last_s,sample['quality'],sample['observed'],sample['four_visible'])
            prediction[j]=exact_rollout(state[:2],state[2:],n,self.last_s,model) if active else prediction[0]
            parameters[j]=model.parameters;disturbance[j]=model.disturbance;allowed[j]=model.update_allowed;updates[j]=model.updates
        return dict(prediction=prediction,parameters=parameters,disturbance=disturbance,allowed=allowed,
                    updates=updates,active=active,normal=n,relative=self.last_s,state=state.copy())

def run_trial(world,scenario,state_mode='estimated',start=FAULT_START,end=FAULT_END,clean_reset=None,retain_reset=False,limit=None):
    dt=world['times'][1]-world['times'][0];bank=ComparisonBank(dt);records=[];reset_snapshot=None;did_reset=False
    for i,t in enumerate(world['times']):
        if limit is not None and t>limit:break
        sample=sensor_sample(world,i,scenario,start,end)
        oracle=np.r_[world['position'][i],world['velocity'][i]] if state_mode=='oracle_current_state' else None
        reset=None
        if clean_reset is not None and not did_reset and t>=end-1e-9:
            reset=clean_reset;did_reset=True
        if retain_reset and reset_snapshot is None and t>=end-1e-9:
            # Snapshot is BEFORE the clean update at this clock time.
            reset_snapshot=copy.deepcopy(bank.models['ordinary_rls'])
        records.append(bank.step(sample,oracle,reset))
    arrays={key:np.asarray([r[key] for r in records]) for key in records[0]}
    arrays['times']=world['times'][:len(records)].copy()
    arrays['truth_parameters']=world['true_parameters'].copy()
    return arrays,reset_snapshot
