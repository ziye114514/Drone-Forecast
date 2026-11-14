"""Explicit thrust-axis dynamics, circle backprojection and causal image adaptation.

Equations (8), (9), (14)-(16) of Thakur et al. motivate the propagation. The
RAW-only image model estimates effective pixel gains, not mass or actual wind.
"""
from collections import deque
import cv2,numpy as np
from scipy.signal import lfilter


def unit(vector):
    x=np.asarray(vector,float)
    return x/max(np.linalg.norm(x),1e-12)


def rotate_normal(normal,omega,dt):
    vector=np.asarray(omega,float)*dt
    angle=np.linalg.norm(vector)
    if angle<1e-12:return unit(normal)
    axis=vector/angle;n=np.asarray(normal,float)
    return unit(n*np.cos(angle)+np.cross(axis,n)*np.sin(angle)+axis*(axis@n)*(1-np.cos(angle)))


def metric_acceleration(omega_rad_s,normal,mass,thrust_coefficient,gravity=(0,0,-9.81)):
    if mass<=0 or thrust_coefficient<0:raise ValueError('Invalid physical parameters')
    thrust=float(thrust_coefficient*np.sum(np.asarray(omega_rad_s,float)**2))
    return thrust/mass*unit(normal)+np.asarray(gravity,float)


def metric_forward(position,velocity,normal,omega_body,rotor_omega,mass,cf,horizon,gravity=(0,0,-9.81),decay=3.):
    p=np.asarray(position,float).copy();v=np.asarray(velocity,float).copy();n=unit(normal);w=np.asarray(omega_body,float).copy()
    count=max(1,int(np.ceil(horizon/.005)));dt=horizon/count
    for _ in range(count):
        middle=rotate_normal(n,w,dt/2)
        a=metric_acceleration(rotor_omega,middle,mass,cf,gravity)
        p+=v*dt+.5*a*dt*dt;v+=a*dt;n=rotate_normal(n,w,dt);w*=np.exp(-decay*dt)
    return p,v,n


def circle_normal_candidates(ellipse,camera_matrix):
    """Two circle-plane normal solutions from the normalized viewing cone.

    This circle-specific conic eigendecomposition is not the full P1E library.
    Signs remain ambiguous until additional orientation/temporal priors enter.
    """
    cx,cy,a,b,angle=ellipse
    c,s=np.cos(angle),np.sin(angle);R=np.array([[c,-s],[s,c]])
    A=R@np.diag([1/a**2,1/b**2])@R.T;center=np.array([cx,cy])
    conic=np.empty((3,3));conic[:2,:2]=A;conic[:2,2]=-A@center;conic[2,:2]=-center@A;conic[2,2]=center@A@center-1
    K=np.asarray(camera_matrix,float);cone=K.T@conic@K
    values,vectors=np.linalg.eigh(cone);order=np.argsort(values)[::-1];values=values[order];vectors=vectors[:,order]
    l1,l2,l3=values
    if not (l1>0 and l2>0 and l3<0):raise ValueError('Ellipse does not define an elliptical viewing cone')
    alpha=np.sqrt(max(0,(l1-l2)/(l1-l3)));beta=np.sqrt(max(0,(l2-l3)/(l1-l3)))
    return [unit(alpha*vectors[:,0]+sign*beta*vectors[:,2]) for sign in (-1,1)]


def weak_normal(rotor,up=(0,-1)):
    angle=np.deg2rad(rotor['angle']);ratio=np.clip(rotor['ratio'],0,1)
    xy=np.array([-np.sin(angle),np.cos(angle)])*np.sqrt(1-ratio*ratio)
    if xy@np.asarray(up)<0:xy=-xy
    return unit(np.r_[xy,ratio])


class ThrustAxisFilter:
    def __init__(self,cfg):
        self.cfg=cfg;self.n=None;self.P=np.eye(3)*.2;self.t=None
        self.omega=np.zeros(3);self.last_measurement=None;self.last_measurement_t=None
        self.differential_gain=0.;self.differential_updates=0;self.quality=0.

    def step(self,t,rotors,differential=0.,enhanced=False):
        dt=t-self.t if self.t is not None else 0.
        if self.n is not None:
            w=self.omega.copy()
            if self.differential_updates>15:w[2]=np.clip(self.differential_gain*differential,-2,2)
            self.n=rotate_normal(self.n,w,dt)
            self.omega*=np.exp(-self.cfg.angular_rate_decay*dt)
            self.P+=np.eye(3)*self.cfg.normal_process_sigma**2*max(dt,0)
        self.t=t
        usable=[r for r in rotors if r.get('geometry_quality',0)>.15]
        if not usable:return self.n
        selected=usable if enhanced else [max(usable,key=lambda r:r.get('activity',1))]
        normals=[];weights=[]
        for r in selected:
            if self.cfg.camera_matrix is not None and not r.get('axis_only',False):
                e=(r['cx'],r['cy'],r['major']/2,r['minor']/2,np.deg2rad(r['angle']))
                if self.cfg.distortion is not None:
                    # Approximate the undistorted rim by sampling the observed
                    # ellipse. This cannot repair a strongly distorted bad fit.
                    phase=np.linspace(0,2*np.pi,96,endpoint=False)
                    c,s=np.cos(e[4]),np.sin(e[4]);R=np.array([[c,-s],[s,c]])
                    rim=np.column_stack((e[2]*np.cos(phase),e[3]*np.sin(phase)))@R.T+np.array(e[:2])
                    K=np.asarray(self.cfg.camera_matrix,float)
                    rim=cv2.undistortPoints(rim.reshape(-1,1,2),K,np.asarray(self.cfg.distortion,float),P=K)
                    center,axes,angle=cv2.fitEllipse(rim.astype(np.float32))
                    e=(*center,axes[0]/2,axes[1]/2,np.deg2rad(angle))
                candidates=circle_normal_candidates(e,self.cfg.camera_matrix)
                choices=[sign*n for n in candidates for sign in (-1,1)]
                up=np.r_[self.cfg.image_up,0.]
                n=max(choices,key=lambda n:float(n@self.n) if self.n is not None else float(n@up))
            else:n=weak_normal(r,self.cfg.image_up)
            if self.n is not None and n@self.n<0:n=-n
            normals.append(n);weights.append(r.get('geometry_quality',.5))
        reference=normals[int(np.argmax(weights))]
        keep=[i for i,n in enumerate(normals) if n@reference>.85] if enhanced else [0]
        measurement=unit(sum(normals[i]*weights[i] for i in keep))
        quality=float(np.mean([weights[i] for i in keep]));self.quality=quality
        if self.n is None:self.n=measurement.copy()
        else:
            R=np.eye(3)*(.025/max(quality,.1))**2
            gain=np.linalg.solve(self.P+R,self.P).T
            self.n=unit(self.n+gain@(measurement-self.n));self.P=(np.eye(3)-gain)@self.P
        if self.last_measurement is not None and t-self.last_measurement_t<.12:
            elapsed=max(t-self.last_measurement_t,.001)
            cross=self.last_measurement[0]*measurement[1]-self.last_measurement[1]*measurement[0]
            signed=np.arctan2(cross,self.last_measurement[:2]@measurement[:2])
            observed_rate=np.clip(signed/elapsed,-3,3)
            self.omega[2]=.8*self.omega[2]+.2*observed_rate
            if abs(differential)>.025 and quality>.45:
                target=np.clip(observed_rate/differential,-15,15)
                self.differential_gain=.995*self.differential_gain+.005*target;self.differential_updates+=1
        self.last_measurement=measurement;self.last_measurement_t=t
        return self.n.copy()


class ProjectedThrustModel:
    def __init__(self,cfg,adaptive=False):
        self.cfg=cfg;self.adaptive=adaptive
        self.parameters=np.array([cfg.nominal_gravity_px_s2,cfg.nominal_gravity_px_s2,0.,0.])
        self.cov=np.diag([.5,.5,.02,.02]);self.queue=deque();self.updates=0
        self.reference_normal=None;self.disturbance=np.zeros(2);self.last_update=None
        self.started=None;self.update_allowed=False

    def acceleration(self,n,s,velocity,include_residual=True):
        if n is None:return np.zeros(2)
        k,g,dx,dy=self.parameters
        a=k*s*np.asarray(n[:2])-g*np.asarray(self.cfg.image_up)-np.array([dx,dy])*velocity
        if self.adaptive and include_residual:a+=self.disturbance
        return np.clip(a,-self.cfg.max_acceleration_px_s2,self.cfg.max_acceleration_px_s2)

    def update(self,t,state,n,s,quality,observed,four_visible):
        if self.started is None:self.started=t
        if self.reference_normal is None and n is not None and four_visible and quality>.35:
            self.reference_normal=np.asarray(n).copy()
            vertical=max(.2,float(np.asarray(n[:2])@np.asarray(self.cfg.image_up)))
            self.parameters[0]=self.parameters[1]/vertical
        self.update_allowed=bool(self.adaptive and observed and four_visible and quality>.45 and n is not None)
        while self.queue and t-self.queue[0][0]>self.cfg.adaptation_delay_s+.06:self.queue.popleft()
        if not self.update_allowed or t-self.started<.6:return
        while self.queue and t-self.queue[0][0]>=self.cfg.adaptation_delay_s:
            old_t,p,v,old_n,old_s,old_q=self.queue.popleft();dt=t-old_t
            target=2*(state[:2]-p-v*dt)/dt**2
            target=np.clip(target,-1500,1500)
            # Four identifiable *effective* pixel coefficients; constraints avoid
            # negative thrust/drag. No claim to recover m, cf or physical wind.
            up=np.asarray(self.cfg.image_up)
            matrix=np.array([[old_s*old_n[0],-up[0],-v[0],0.],
                             [old_s*old_n[1],-up[1],0.,-v[1]]])
            scaled=matrix*np.array([200.,200.,1.,1.])
            theta=self.parameters/np.array([200.,200.,1.,1.])
            error=np.clip(target/200.-matrix@self.parameters/200.,-2,2)
            X=scaled/200.;S=np.eye(2)*(.5/max(old_q,.1))+X@self.cov@X.T
            gain=np.linalg.solve(S,X@self.cov).T
            # A single almost-hover view cannot separate thrust from gravity.
            excitation=np.linalg.norm(old_n[:2]-self.reference_normal[:2])+abs(old_s-1)
            if excitation<.035:gain[:2]=0
            theta+=gain@error
            self.parameters=theta*np.array([200.,200.,1.,1.])
            self.parameters[:2]=np.clip(self.parameters[:2],20.,5000.)
            self.parameters[2:]=np.clip(self.parameters[2:],0.,3.)
            self.cov=(self.cov-gain@X@self.cov)/self.cfg.adaptation_forgetting
            self.cov=(self.cov+self.cov.T)/2
            values,vectors=np.linalg.eigh(self.cov);self.cov=vectors@np.diag(np.clip(values,1e-6,100))@vectors.T
            residual=target-self.acceleration(old_n,old_s,v,False)
            self.disturbance=.97*self.disturbance+.03*np.clip(residual,-300,300)
            self.last_update=t;self.updates+=1
        self.queue.append((t,state[:2].copy(),state[2:4].copy(),np.asarray(n).copy(),s,quality))

    def rollout(self,position,velocity,normal,omega,s,horizons=None):
        times=np.linspace(0,.5,26) if horizons is None else np.asarray(horizons)
        p=np.asarray(position,float);v=np.asarray(velocity,float);n=unit(normal);w=np.asarray(omega,float)
        count=max(1,int(np.ceil(times[-1]/.005)));dt=times[-1]/count
        middle=(np.arange(count)+.5)*dt;speed=np.linalg.norm(w)
        if speed>1e-10:
            axis=w/speed;beta=self.cfg.angular_rate_decay
            angles=speed*(-np.expm1(-beta*middle)/beta if beta>0 else middle)
            normals=(np.cos(angles)[:,None]*n+np.sin(angles)[:,None]*np.cross(axis,n)
                     +(1-np.cos(angles))[:,None]*axis*(axis@n))
        else:normals=np.tile(n,(count,1))
        k,g,dx,dy=self.parameters
        forcing=k*s*normals[:,:2]-g*np.asarray(self.cfg.image_up)
        if self.adaptive:forcing+=self.disturbance
        forcing=np.clip(forcing,-self.cfg.max_acceleration_px_s2,self.cfg.max_acceleration_px_s2)
        positions=np.empty((count+1,2));positions[0]=p
        for j,drag in enumerate((dx,dy)):
            if drag<1e-8:
                velocities=v[j]+np.cumsum(forcing[:,j])*dt
                previous=np.r_[v[j],velocities[:-1]]
                positions[1:,j]=p[j]+np.cumsum(previous*dt+.5*forcing[:,j]*dt**2)
            else:
                retention=np.exp(-drag*dt);input_gain=-np.expm1(-drag*dt)/drag
                velocities,_=lfilter([input_gain],[1,-retention],forcing[:,j],zi=[retention*v[j]])
                previous=np.r_[v[j],velocities[:-1]]
                positions[1:,j]=p[j]+np.cumsum(input_gain*previous+(dt-input_gain)/drag*forcing[:,j])
        internal=np.linspace(0,times[-1],count+1)
        return times,np.column_stack([np.interp(times,internal,positions[:,j]) for j in range(2)])
