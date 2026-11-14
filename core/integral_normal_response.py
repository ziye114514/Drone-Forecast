"""Estimate unknown collective normal response from integrated native motion.

Delta v = A*integral(n) + b*Delta t - beta*integral(v). No rotor frequency
is invented: A belongs to the UNOBSERVED response. External b is eliminated
without a zero-wind/gravity prior; beta retains the existing weak response
prior. Only separated native displacement endpoints form each factor.
Current p/v/a is unchanged. The fit only refines a qualified future normal
drive. Image units and perspective approximation prevent metric-force claims.
"""
from collections import deque
import copy
import numpy as np
from axis_amplitude_bridge import current_axis_qualified


def integrated_axis(history,start,end):
    pairs=list(history)
    if len(pairs)<2 or pairs[0][0]>start or pairs[-1][0]<end:return None
    value=np.zeros(2)
    for (a,n),(b,m) in zip(pairs[:-1],pairs[1:]):
        lo=max(start,a);hi=min(end,b)
        if hi<=lo:continue
        if b-a>.12+1e-8:return None
        u=(lo-a)/(b-a);v=(hi-a)/(b-a)
        value+=(hi-lo)*((1-u)*n+u*m+(1-v)*n+v*m)/2
    return value


def fit_response(samples,min_factors=8):
    if len(samples)<min_factors:return dict(qualified=False,reason='insufficient_disjoint_native_motion_factors',factors=len(samples))
    design=[];target=[]
    for _,dn,dv,dt,y,R in samples:
        L=np.linalg.cholesky(R);X=np.column_stack((dn,np.array([[dt,0],[0,dt]]),-dv))
        design.append(np.linalg.solve(L,X));target.append(np.linalg.solve(L,y))
    X=np.vstack(design);y=np.concatenate(target);B=X[:,1:3]
    inv=np.linalg.pinv(B);Q=X[:,[0,3]]-B@(inv@X[:,[0,3]]);z=y-B@(inv@y)
    # External2D force has no finite zero-mean prior. beta's weak prior is
    # the old1/.35 +/-1/.35; A has no artificial finite prior in this fit.
    beta=1/.35;precision=Q.T@Q+np.diag([0.,1/beta**2]);rhs=Q.T@z+np.array([0.,1/beta])
    if np.linalg.matrix_rank(precision)<2:return dict(qualified=False,reason='normal_response_not_excited',factors=len(samples))
    C=np.linalg.inv(precision);p=C@rhs;residual=z-Q@p
    scale=max(1.,float(residual@residual)/max(len(z)-4,1));C*=scale
    variance=float(C[0,0]);amplitude=float(p[0])
    if variance<=0 or not np.isfinite(variance+amplitude):return dict(qualified=False,reason='degenerate_response_fit')
    # Same1000px/s² weak unknown-response scale as the inherited observer.
    # This is a dependence-aware conservative blend proxy, not a posterior
    # independence/probability claim about these shared native observations.
    weight=1e6/(1e6+variance) if amplitude>0 else 0.
    return dict(qualified=bool(weight>0),reason='past_integrated_normal_excitation' if weight>0 else 'positive_normal_response_unsupported',
        factors=len(samples),amplitude=amplitude,sigma_proxy=float(np.sqrt(variance)),weight=float(weight),
        beta_nuisance=float(p[1]),external_nuisance_rank=int(np.linalg.matrix_rank(B)),
        residual_noise_scale=scale,fit_uses_weak_beta_prior=True,metric_force_identified=False)


class IntegralNormalResponse:
    def __init__(self,span_s=.1,min_factors=8):
        self.span_s=span_s;self.min_factors=min_factors
        self.normals=deque();self.factors=deque();self.anchor=None;self.previous=None;self.integral_v=np.zeros(2);self.skip=False;self.observations=None

    def step(self,row):
        t=row['t'];obs=row.get('state',{}).get('observations')
        if obs is not None and self.observations is not None and obs<self.observations:self.__init__(self.span_s,self.min_factors)
        self.observations=obs
        st=row['structure'];mo=row['motion'];s=mo.get('estimate') or {};dt=mo.get('dt_s')
        n=np.asarray(st.get('axis_xy') or [],float)
        ngood=bool(current_axis_qualified(st) and n.shape==(2,) and np.linalg.norm(n)>1e-6)
        if ngood:
            if n[1]>0:n=-n
            self.normals.append((t-.01,n.copy()))
        while self.normals and t-self.normals[0][0]>3.3:self.normals.popleft()
        good=bool(ngood and dt and 0<dt<=.12 and mo.get('current_observation') and s.get('score',0)>0 and s.get('current_displacement_qualified',True) and not s.get('peak_at_search_boundary'))
        if good:
            velocity=np.array([s['dx'],s['dy']])/dt
            variance=np.asarray(s.get('displacement_covariance_px2',np.eye(2)*4.),float)/dt**2
            vt=t-dt/2-float(mo.get('native_motion_accumulation_us',33333))*1e-6/2
            previous=self.previous;sample=(vt,velocity,variance,t,dt)
            if previous is not None and abs((t-previous[3])-dt)>1e-6:self.anchor=None;self.integral_v[:]=0
            if self.anchor is None:
                if self.skip:self.skip=False
                else:self.anchor=sample;self.integral_v[:]=0
            elif previous is not None:
                self.integral_v+=(velocity+previous[1])*(vt-previous[0])/2
                span=vt-self.anchor[0]
                if span>=self.span_s-1e-7:
                    dn=integrated_axis(self.normals,self.anchor[0],vt)
                    if dn is not None:
                        # Tangential frame errors are proxies; do not divide
                        # by event/feature counts or count overlapping pairs.
                        R=variance+self.anchor[2]+np.eye(2)*(1000*np.deg2rad(5)*span)**2
                        self.factors.append((t,dn,self.integral_v.copy(),span,velocity-self.anchor[1],R))
                    self.anchor=None;self.integral_v[:]=0;self.skip=True
            self.previous=sample
        else:self.previous=None;self.anchor=None;self.integral_v[:]=0;self.skip=False
        while self.factors and t-self.factors[0][0]>3.:self.factors.popleft()
        info=fit_response(self.factors,self.min_factors);r=copy.deepcopy(row);used=[]
        if info.get('qualified') and ngood and row['coefficient']:
            def apply(packet):
                f=packet['force'];normal=packet.get('normal_turn_forecast') or {}
                if normal.get('applied') and f.get('normal_turn_vector_px_s2') is not None:
                    old=np.asarray(f['normal_turn_vector_px_s2']);w=info['weight'];new=(1-w)*old+w*info['amplitude']*n
                    f['normal_turn_vector_px_s2']=new.tolist();f['integral_unknown_normal_response']=dict(info,current_pva_unchanged=True,hidden_rotor_force_measured=False)
                    used.append(bool(np.any(new!=old)))
                for part in f.get('response_model_mixture',[]):apply(part)
            apply(r)
        r['integral_normal_response']=dict(info,applied=bool(any(used)),current_state_unchanged=True,frequency_inputs_unchanged=True)
        return r
