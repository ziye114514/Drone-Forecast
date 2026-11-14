"""Coupled thrust-axis and driven position filters with explicit future rollout."""
from collections import deque
import numpy as np
from .config import Config
from .motion import ForecastBank,extrapolate
from .paper_physics import ThrustAxisFilter,ProjectedThrustModel


class DrivenPositionFilter:
    def __init__(self):
        self.x=None;self.P=np.diag([49.,49.,1e5,1e5]);self.t=None

    def step(self,t,measurement,acceleration,confidence):
        if self.x is None:
            if measurement is None:return None
            self.x=np.r_[measurement,0.,0.];self.t=t;return self.x.copy()
        dt=t-self.t;F=np.eye(4);F[:2,2:]=np.eye(2)*dt
        B=np.vstack((np.eye(2)*dt**2/2,np.eye(2)*dt))
        self.x=F@self.x+B@acceleration;self.P=F@self.P@F.T+B@B.T*1800**2;self.t=t
        if measurement is not None:
            H=np.column_stack((np.eye(2),np.zeros((2,2))));R=np.eye(2)*(7/max(confidence,.25))**2
            gain=np.linalg.solve(H@self.P@H.T+R,H@self.P).T
            self.x+=gain@(np.asarray(measurement)-self.x[:2])
            A=np.eye(4)-gain@H;self.P=A@self.P@A.T+gain@R@gain.T
        return self.x.copy()


class PaperForecastBank:
    def __init__(self,cfg):
        self.cfg=cfg
        base=Config(step_us=cfg.step_us,opencv_threads=cfg.opencv_threads,horizons=cfg.horizons)
        self.base=ForecastBank(base)
        self.axes={name:ThrustAxisFilter(cfg) for name in ('paper_nominal','paper_adaptive')}
        self.models={name:ProjectedThrustModel(cfg,name=='paper_adaptive') for name in self.axes}
        self.filters={name:DrivenPositionFilter() for name in self.axes}
        self.previous_a={name:np.zeros(2) for name in self.axes}
        self.weights=np.array([.9,.05,.05]);self.completed=deque();self.last_seen=None;self.last_force_t=None
        self.last_ratio=1.;self.last_quality=0.;self.last_rotors=[]

    def step(self,observation,sensing):
        t=observation['t'];observed=observation.get('detected',False)
        if self.last_seen is not None and t-self.last_seen>.5:
            self.__init__(self.cfg)
        if observed:self.last_seen=t
        result=self.base.step(observation)
        if result is None:return None
        z=observation['center'] if observed else None
        valid_rotors=[r for r in sensing['rotors'] if r['observed_now']]
        if sensing['four_frequencies'] and sensing['thrust_ratio'] is not None:
            self.last_ratio=float(np.clip(sensing['thrust_ratio'],.2,5.));self.last_force_t=t
        signal_fresh=self.last_force_t is not None and t-self.last_force_t<=.16
        four_visible=sensing['four_visible']
        quality=sensing['frequency_quality']*observation.get('geometry_quality',0)
        active=bool(signal_fresh and quality>.15)
        states=result['states']
        for name,axis in self.axes.items():
            normal=axis.step(t,valid_rotors,sensing['differential_proxy'],name=='paper_adaptive')
            state=self.filters[name].step(t,z,self.previous_a[name],observation.get('confidence',0))
            model=self.models[name]
            model.update(t,self.base.cv.x,normal,self.last_ratio,quality,observed,four_visible)
            if active and normal is not None:
                a=model.acceleration(normal,self.last_ratio,state[2:])
                times,points=model.rollout(state[:2],state[2:],normal,axis.omega,self.last_ratio)
                states[name]=dict(position=state[:2].tolist(),velocity=state[2:].tolist(),acceleration=a.tolist(),
                                  rollout_times=times.tolist(),rollout_positions=points.tolist())
            else:
                a=np.zeros(2);states[name]=dict(states['cv'])
            self.previous_a[name]=a
        names=('cv','paper_nominal','paper_adaptive')
        while self.completed and t-self.completed[0][0]>.26:self.completed.popleft()
        if observed and observation.get('confidence',0)>.5:
            while self.completed and t-self.completed[0][0]>=.2:
                old_t,old_states=self.completed.popleft();elapsed=t-old_t
                loss=np.array([np.linalg.norm(extrapolate(old_states[n],elapsed)-np.asarray(z))**2 for n in names])
                loss=np.clip(loss/35**2,0,16)
                self.weights*=np.exp(-.08*loss);self.weights/=max(self.weights.sum(),1e-12)
                self.weights=.99*self.weights+.01/3
            self.completed.append((t,{name:dict(states[name]) for name in names}))
        weights=self.weights.copy() if active else np.array([1.,0,0])
        times=np.linspace(0,.5,26)
        curves=[]
        for name in names:
            state=states[name]
            if 'rollout_positions' in state:curves.append(np.asarray(state['rollout_positions']))
            else:curves.append(np.asarray(state['position'])+times[:,None]*state['velocity']+.5*times[:,None]**2*state['acceleration'])
        points=sum(weights[i]*curves[i] for i in range(3))
        states['paper_fused']={key:sum(weights[i]*np.array(states[name][key]) for i,name in enumerate(names)).tolist()
                               for key in ('position','velocity','acceleration')}
        states['paper_fused'].update(rollout_times=times.tolist(),rollout_positions=points.tolist())
        result.update(physics_active=active,four_visible=four_visible,four_frequencies=sensing['four_frequencies'],
            thrust_ratio=self.last_ratio,thrust_axis={name:axis.n.tolist() if axis.n is not None else None for name,axis in self.axes.items()},
            angular_velocity_proxy={name:axis.omega.tolist() for name,axis in self.axes.items()},
            image_differential_gain={name:axis.differential_gain for name,axis in self.axes.items()},
            physical_parameters={name:dict(thrust_gain_px_s2=float(m.parameters[0]),gravity_gain_px_s2=float(m.parameters[1]),
                drag_per_s=m.parameters[2:].tolist(),updates=m.updates,update_allowed=m.update_allowed,
                residual_px_s2=m.disturbance.tolist()) for name,m in self.models.items()},
            physical_model_weights=dict(zip(names,weights.tolist())),
            interpretation='RAW-only projected physical adaptation; pixel gains are not verified m, cf or wind. Future RPM is held, angular rate decays.')
        return result
