"""Image-force observer with a geometry-conditioned UNKNOWN thrust component.

a = k*u_observed + d_unknown; internally d_unknown = b*n + e.
b is unmeasured, not a fabricated hidden RPM. e includes other unknown effects.
The n-dependent prior is testable and optional; no Newton force is identified.
"""
import numpy as np


class AxisUnknownObserver:
    correlation_s=.35

    def __init__(self):
        self.x=self.P=None;self.t=self.seen=None;self.observations=0
        self.u=np.zeros(2);self.n=np.zeros(2);self.components={}

    def step(self,t,body,u,quality=0.,allow_gain_learning=False,input_meta=None,structure=None):
        if self.seen is not None and t-self.seen>.5:self.__init__()
        z=np.asarray(body['center'],float) if body.get('detected') else None
        raw=(structure or {}).get('axis_xy');current_axis=False
        n=self.n.copy()
        if z is not None and raw is not None:
            value=np.asarray(raw,float)
            if value.shape==(2,) and np.isfinite(value).all() and np.linalg.norm(value)>1e-6:
                n=-value if value[1]>0 else value;current_axis=True
        if self.x is None:
            if z is None:return None,False,dict(active=False,acceleration_px_s2=[0.,0.]),dict(reason='uninitialized')
            self.x=np.r_[z,np.zeros(2),-2000.*n,200.,2000.]
            self.P=np.diag([10000.]*4+[1000000.]*2+[40000.,4000000.])
            # A broad amplitude prior cancels in the initial net unknown force.
            J=np.eye(8);J[4:6,7]=-n;self.P=J@self.P@J.T
            self.n=n.copy();dt=0.
        else:
            dt=t-self.t
            if dt<=0:raise ValueError('Increasing source timestamps required')
            F=np.eye(8);F[:2,2:4]=np.eye(2)*dt;F[:2,4:6]=np.eye(2)*(.5*dt*dt);F[2:4,4:6]=np.eye(2)*dt
            F[:2,6]=.5*dt*dt*self.u;F[2:4,6]=dt*self.u
            F[:2,7]=.5*dt*dt*self.n;F[2:4,7]=dt*self.n
            base=np.array([[dt**5/20,dt**4/8,dt**3/6],[dt**4/8,dt**3/3,dt**2/2],[dt**3/6,dt**2/2,dt]])
            Q=np.zeros((8,8));env_sigma=100. if np.linalg.norm(self.n)>0 else 1000.
            for axis in range(2):
                ix=[axis,axis+2,axis+4];Q[np.ix_(ix,ix)]+=base*(2*env_sigma**2/.35)
            G=np.zeros((8,3));G[:2,0]=self.n;G[2:4,1]=self.n;G[7,2]=1.
            Q+=G@base@G.T*(2*1000.**2/.35)
            self.x=F@self.x;self.P=F@self.P@F.T+Q
        self.t=float(t)
        if z is not None:self.observations+=1;self.seen=t
        valid=bool(self.observations>=5 and self.seen is not None and t-self.seen<=.16)
        innovation=None
        if z is not None:
            H=np.zeros((2,8));H[:,:2]=np.eye(2);R=np.eye(2)*36.
            residual=z-self.x[:2];S=self.P[:2,:2]+R;K=np.linalg.solve(S,self.P[:2,:]).T
            K[6]=0. # Keep the old effective observed-input scale for this ablation.
            innovation=float(residual@np.linalg.solve(S,residual));self.x+=K@residual
            J=np.eye(8)-K@H;self.P=J@self.P@J.T+K@R@K.T
        good=bool(valid and u is not None and quality>=.35 and np.isfinite(u).all())
        meta=input_meta or {};new_components={str(k):np.asarray(v,float) for k,v in meta.get('observed_components',{}).items()} if good else {}
        new_u=np.asarray(u,float) if good else np.zeros(2)
        if good and not np.allclose(np.sum(list(new_components.values()),axis=0),new_u):raise ValueError('Contribution accounting mismatch')
        old_keys=set(self.components);new_keys=set(new_components);delta=np.zeros(2)
        for k in new_keys-old_keys:delta+=new_components[k]
        for k in old_keys-new_keys:delta-=self.components[k]
        # Repartition current observed contribution only; real changes in
        # retained inputs/normals must not be canceled by bookkeeping.
        J=np.eye(8);J[4:6,6]=-delta;self.x[4:6]-=self.x[6]*delta;self.P=J@self.P@J.T
        if current_axis and np.linalg.norm(self.n)==0:
            self.x[4:6]+=self.x[7]*(self.n-n)
            J=np.eye(8);J[4:6,7]=self.n-n;self.P=J@self.P@J.T
        self.u=new_u;self.n=n;self.components=new_components
        constrained=self.x[7]<0
        if constrained:
            # A nonphysical negative amplitude is moved into the unstructured
            # unknown term, preserving the estimated net acceleration.
            self.x[4:6]+=self.x[7]*self.n;self.x[7]=0.
            J=np.eye(8);J[4:6,7]=self.n;J[7,:]=0.;self.P=J@self.P@J.T
            self.P[7,7]=1000000.
        self.P=.5*(self.P+self.P.T)
        unknown=self.x[4:6]+self.x[7]*self.n;a=self.x[6]*self.u+unknown
        norm=np.linalg.norm(a)
        if norm>3000:a=a*(3000./norm)
        c=dict(position=self.x[:2].tolist(),velocity=self.x[2:4].tolist(),acceleration=a.tolist())
        f=dict(active=good,effective_gain_px_s2=float(self.x[6]),input_u=self.u.tolist() if good else None,
            disturbance_px_s2=unknown.tolist(),acceleration_px_s2=a.tolist(),acceleration_correlation_s=self.correlation_s,
            disturbance_estimator='joint_position_axis_conditioned_unknown_force_KF',
            unobserved_axis_amplitude_proxy=float(self.x[7]),unobserved_axis_amplitude_sigma=float(np.sqrt(max(0.,self.P[7,7]))),
            unobserved_axis_force_is_measured=False,other_unknown_residual_px_s2=self.x[4:6].tolist(),
            current_axis_observed=current_axis,negative_amplitude_repartitioned=bool(constrained),
            gain_learned=False,gain_ever_learned=False,metric_force_measured=False,
            force_decomposition='observed_proxy_plus_axis_conditioned_unknown_force',
            hidden_rotor_force_owner='unknown_b_times_n_plus_unknown_residual',
            acceleration_source='observed_proxy_plus_geometry_conditioned_unknown_prior',future_commands_observed=False)
        state=dict(observations=self.observations,measurement_innovation_chi2=innovation,
            joint_position_velocity_disturbance=True,parallel_trajectory_branch=False,
            position_sigma_px=np.sqrt(np.maximum(0.,np.diag(self.P)[:2])).tolist())
        return c,valid,f,state
