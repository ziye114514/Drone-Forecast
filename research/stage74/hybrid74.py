"""Actual author IMM core: parallel box CV and controlled maneuver model.

Each hypothesis gets the same position observation once. Covariance and mode
interaction follow FilterPy. Effective force inputs are hypotheses, not extra
independent position measurements. Original33 predictions remain available.
"""
from pathlib import Path
import copy,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage69'),str(ROOT/'research/stage14')]
from force69 import Direction,prior
VENDOR=ROOT/'external/filterpy_imm_upstream'
snapshot=json.loads((VENDOR/'UPSTREAM_SNAPSHOT.json').read_text('utf-8'))
for item in snapshot['files']:
    data=(VENDOR/item['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==item['sha256']
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['git_blob_sha1']
_source=(VENDOR/'IMM.py').read_text('utf-8')
_old='from filterpy.common import pretty_str'
assert _source.count(_old)==1
_ns=dict(__name__='filterpy_imm74')
exec(compile(_source.replace(_old,"def pretty_str(name, value): return str(name)+' = '+str(value)"),str(VENDOR/'IMM.py'),'exec'),_ns)
IMMEstimator=_ns['IMMEstimator']
ADDED=['dual_cv_agile_zero','dual_cv_history','dual_cv_axis_constant','dual_cv_event_force','dual_cv_increment_constant','dual_cv_event_increment']
PARAMS=dict(initial_mode_probability=[.9,.1],transition=[[.97,.03],[.03,.97]],
    stable_sigma_acceleration=400.,agile_sigma_acceleration=800.,position_sigma=6.,
    physics_min_updates=10,acceleration_cap=2500.,reset_gap_s=.5,
    likelihood='Current innovation, full Gaussian covariance, underflow floor exp(-700). '
               'No .2/.4s wait for mode evidence; causally available force input only.',
    state='Both modes4D [x,y,vx,vy]; original author IMM interaction and between-mode covariance.',
    controls='agile_zero isolates higher process noise; history isolates kinematic acceleration; '
             'axis_constant and increment_constant are matched event-admission controls.',
    interpretation='Image-plane effective force and approximate uncertainty, not verified metric dynamics. '
                   'Force parameters/gain used for control come from the preceding tick, '
                   'not the current position feedback being scored by mode likelihood.')


class ControlledFilter:
    def __init__(self,position):
        self.x=np.r_[position,[0.,0.]].astype(float);self.P=np.eye(4)*1e4
        self.dt=.02;self.a=np.zeros(2);self.sigma=400.;self.likelihood=1.
        self.innovation=np.zeros(2);self.nis=0.
    def predict(self,u=None):
        dt=self.dt;F=np.eye(4);F[0,2]=dt;F[1,3]=dt
        B=np.r_[np.eye(2)*(.5*dt*dt),np.eye(2)*dt]
        self.x=F@self.x+B@self.a;self.P=F@self.P@F.T+self.sigma**2*(B@B.T)
    def update(self,z):
        if z is None:self.likelihood=1.;return
        H=np.c_[np.eye(2),np.zeros((2,2))];R=np.eye(2)*36.
        y=np.asarray(z)-H@self.x;S=H@self.P@H.T+R
        K=np.linalg.solve(S,H@self.P).T;self.innovation=y;self.nis=float(y@np.linalg.solve(S,y))
        sign,logdet=np.linalg.slogdet(S);assert sign>0
        self.likelihood=float(np.exp(max(-700.,-.5*(self.nis+logdet+2*np.log(2*np.pi)))))
        self.x+=K@y;A=np.eye(4)-K@H;self.P=A@self.P@A.T+K@R@K.T;self.P=(self.P+self.P.T)/2


class DualBank:
    def __init__(self):self.imm=None;self.last_t=None
    def step(self,t,z,acceleration,agile):
        if self.imm is None:
            if z is None:return
            filters=[ControlledFilter(z),ControlledFilter(z)]
            self.imm=IMMEstimator(filters,[.9,.1],np.array([[.97,.03],[.03,.97]]));self.last_t=t
            return
        dt=t-self.last_t;assert dt>0;self.last_t=t
        for index,f in enumerate(self.imm.filters):
            f.dt=dt;f.a=np.zeros(2) if index==0 else np.clip(acceleration,-2500.,2500.)
            f.sigma=800. if index==1 and agile else 400.
        self.imm.predict()
        if z is not None:self.imm.update(np.asarray(z,float))
        assert np.isfinite(self.imm.x).all() and np.isfinite(self.imm.P).all()
        assert abs(self.imm.mu.sum()-1)<1e-12
        assert np.linalg.eigvalsh(self.imm.P).min()>-1e-7
    def coefficient(self):
        times=np.linspace(0,.5,26);points=[];uncertainty=[]
        for h in times:
            F=np.eye(4);F[0,2]=h;F[1,3]=h;B=np.r_[np.eye(2)*(.5*h*h),np.eye(2)*h]
            means=[(F@f.x+B@f.a)[:2] for f in self.imm.filters]
            p=sum(mu*x for mu,x in zip(self.imm.mu,means));points.append(p)
            covariance=np.zeros((2,2))
            for mu,f,x in zip(self.imm.mu,self.imm.filters,means):
                C=(F@f.P@F.T+f.sigma**2*(B@B.T))[:2,:2];e=x-p
                covariance+=mu*(C+np.outer(e,e))
            uncertainty.append(covariance.tolist())
        c=dict(position=self.imm.x[:2].tolist(),velocity=self.imm.x[2:].tolist(),
            acceleration=sum(mu*f.a for mu,f in zip(self.imm.mu,self.imm.filters)).tolist(),
            rollout_times=times.tolist(),rollout_positions=np.asarray(points).tolist())
        d=dict(mode_probability=self.imm.mu.tolist(),innovation_nis=[f.nis for f in self.imm.filters],
            component_acceleration=[f.a.tolist() for f in self.imm.filters],position_covariance=self.imm.P[:2,:2].tolist(),
            rollout_position_covariance=uncertainty)
        return c,d


class HybridForecast:
    def __init__(self):self.reset()
    def reset(self):
        self.models={m:DualBank() for m in ADDED};self.direction=Direction();self.last_seen=None
        self.previous_force={};self.previous_gain={};self.previous_history_a=np.zeros(2)
    def step(self,original,obs,body,rotor_quality):
        q=copy.deepcopy(original);t=q['t']
        if self.last_seen is not None and t-self.last_seen>.5:self.reset()
        z=obs.get('center') if obs.get('detected') else None
        if z is not None:self.last_seen=t
        n,nd=self.direction.step(t,q,body,rotor_quality);old_force=q['event_force_diagnostic']['models'];inc=q['increment_diagnostic']['models']
        diagnostics={}
        for name,model in self.models.items():
            acceleration=np.zeros(2);active=False
            if name=='dual_cv_agile_zero':active=True
            elif name=='dual_cv_history':acceleration=self.previous_history_a.copy();active=True
            elif name in ['dual_cv_axis_constant','dual_cv_event_force']:
                key='force_periodic_constant' if name.endswith('constant') else 'force_periodic'
                d=old_force[key];old=self.previous_force.get(key)
                active=bool(n is not None and d['active'] and old and old['updates']>=10)
                if active:
                    k,g,dx,dy=old['effective_parameters'];v=model.imm.x[2:] if model.imm is not None else np.zeros(2)
                    acceleration=k*d['ratio']*n-g*np.array([0.,-1.])-np.array([dx,dy])*v+np.asarray(old['disturbance'])
            else:
                key='increment_periodic_constant' if name.endswith('constant') else 'increment_periodic'
                d=inc[key];gain=self.previous_gain.get(key)
                active=bool(d['active'] and gain is not None and gain['updates']>=10 and d['delta'] is not None)
                if active:acceleration=self.previous_history_a+gain['gain_px_s2']*np.asarray(d['delta'])
            model.step(t,z,acceleration,active)
            if model.imm is not None:
                c,d=model.coefficient();d.update(physical_input_active=active,control_acceleration=np.asarray(acceleration).tolist())
                if q['valid']:
                    q['coefficients'][name]=c;q['predictions'][name]={str(h):prior.evaluate(c,h) for h in prior.PARAMS['horizons']}
            else:d=dict(physical_input_active=False,mode_probability=None)
            diagnostics[name]=d
        self.previous_force=copy.deepcopy(old_force);self.previous_gain=copy.deepcopy(inc)
        if q['valid']:self.previous_history_a=np.asarray(q['coefficients']['trusted_acc']['acceleration'],float)
        q['dual_branch_diagnostic']=diagnostics
        return q


def controls():
    # True force is supplied as a controlled sensor input here. Compare missing,
    # reversed and160ms-delayed versions to show dependence on cue correctness.
    t=np.arange(0,3.0001,.02);a=np.zeros((len(t),2));a[(t>=1)&(t<1.5)]=[0.,800.];a[(t>=2)&(t<2.5)]=[-600.,0.]
    p=np.zeros_like(a);v=np.zeros_like(a);v[0]=[150.,0.]
    for j in range(1,len(t)):p[j]=p[j-1]+v[j-1]*.02+.5*a[j-1]*.02**2;v[j]=v[j-1]+a[j-1]*.02
    records=[]
    for kind in ['correct_force','missing_force','reversed_force','delayed_force']:
        bank=DualBank();saved=[];rng=np.random.default_rng(7400)
        for j,ts in enumerate(t):
            u=a[j] if kind=='correct_force' else np.zeros(2) if kind=='missing_force' else -a[j] if kind=='reversed_force' else a[max(0,j-8)]
            bank.step(float(ts),(p[j]+rng.normal(0,6,2)).tolist(),u,kind!='missing_force')
            c,d=bank.coefficient();saved.append((c,d))
        for region,mask in [('steady',(t>=.5)&(t<.8)),('maneuver',(t>=1)&(t<=1.28))]:
            error=[float(np.linalg.norm(np.asarray(prior.evaluate(saved[j][0],.2))-p[j+10])) for j in np.flatnonzero(mask)]
            records.append(dict(kind=kind,region=region,n=len(error),mean_fde_px=float(np.mean(error))))
    # Author interaction matrix must be normalised, and PSD covariance holds.
    assert np.allclose(bank.imm.omega.sum(axis=0),1.)
    return dict(table=records,author_mixing_probabilities_normalized=True,finite_psd_covariance=True,
        scope='Known synthetic positions/acceleration, not event-sensor simulation. '
              'Correct input is an oracle cue for mechanism testing only; RAW tests remain decisive.')
