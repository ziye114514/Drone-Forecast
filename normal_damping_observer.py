"""Normal-conditioned unknown force with a passive image-damping response.

Retain a=k*u+d and d=e+h*n-lambda*v. Scalar lambda is an image-response
parameter learned causally, NOT measured mechanical drag or wind velocity.
With constant n/u and constant external e, net acceleration obeys da=-lambda*a.
An affine-constant state permits exact EKF mean and analytic lambda Jacobian.
"""
import numpy as np
from normal_response_observer import NormalResponseObserver


def kernels(rate,dt):
    z=rate*dt
    if abs(z)<1e-4:
        q1=dt-rate*dt**2/2+rate**2*dt**3/6-rate**3*dt**4/24
        q2=dt**2/2-rate*dt**3/6+rate**2*dt**4/24-rate**3*dt**5/120
        d1=-dt**2/2+rate*dt**3/3-rate**2*dt**4/8
        d2=-dt**3/6+rate*dt**4/12-rate**2*dt**5/40
    else:
        q1=-np.expm1(-z)/rate;q2=(dt-q1)/rate
        d1=(z*np.exp(-z)+np.expm1(-z))/rate**2;d2=(-rate*d1-dt+q1)/rate**2
    return q1,q2,np.exp(-z),d1,d2


class NormalDampingObserver(NormalResponseObserver):
    state_dim=14
    damping_index=12
    constant_index=13
    def _initialize(self,z):
        super()._initialize(z);self.x[self.constant_index]=1.
        self.P[self.damping_index,self.damping_index]=(1/self.correlation_s)**2

    def _transition(self,dt):
        damping=max(0.,float(self.x[self.damping_index]));rate=damping+(1/self.correlation_s if self.consistent_dynamics else 0.)
        q1,q2,phi,d1,d2=kernels(rate,dt);a=self.x[4:6]+self.x[self.gain_index]*self.u
        F=np.eye(self.state_dim);F[:2,2:4]=np.eye(2)*dt;F[:2,4:6]=np.eye(2)*q2;F[2:4,4:6]=np.eye(2)*q1;F[4:6,4:6]=np.eye(2)*phi
        F[:2,self.gain_index]=q2*self.u;F[2:4,self.gain_index]=q1*self.u;F[4:6,self.gain_index]=(phi-1)*self.u
        derivative=np.r_[d2*a,d1*a,-dt*phi*a]
        F[:6,self.damping_index]=derivative;F[:6,self.constant_index]=-derivative*damping
        # Same disturbance spectral density as191; integrate its damped input
        # response with positive Gauss-Legendre weights to preserve PSD.
        nodes,weights=np.polynomial.legendre.leggauss(8);s=(nodes+1)*dt/2;weights=weights*dt/2
        vv=np.array([[kernels(rate,t)[1],kernels(rate,t)[0],np.exp(-rate*t)] for t in s])
        block=vv.T@(weights[:,None]*vv)*(2*1000.**2/self.correlation_s)
        Q=np.zeros((self.state_dim,self.state_dim));Q[:6,:6]=np.kron(block,np.eye(2))
        return F,Q

    def _input_covariance_map(self,F):
        B=super()._input_covariance_map(F);B[4:6]=np.eye(2)*(F[4,4]-1.);return B

    def step(self,t,body,u,*args,**kwargs):
        was_two=self.two_point_initialized;c,valid,f,s=super().step(t,body,u,*args,**kwargs)
        if c is not None:
            if self.two_point_initialized and not was_two:self.P[self.damping_index,self.damping_index]=(1/self.correlation_s)**2
            # Passive response and existing0.1s fastest gap prior supply bounds.
            self.x[self.damping_index]=np.clip(self.x[self.damping_index],0.,10.)
            rate=float(self.x[self.damping_index]);f['acceleration_correlation_s']=1/(1/f['acceleration_correlation_s']+rate)
            drag=-rate*self.x[2:4]
            f.update(image_damping_rate_s_inv=rate,image_damping_sigma_s_inv=float(np.sqrt(max(0.,self.P[self.damping_index,self.damping_index]))),
                image_damping_acceleration_px_s2=drag.tolist(),mechanical_drag_measured=False,
                future_net_acceleration_prior='command_uncertainty_plus_learned_image_damping',
                residual_external_before_image_damping=(self.x[4:6]-self.x[self.response_index]*(np.zeros(2) if self.normal is None else self.normal)-drag).tolist())
        s['passive_damping_starts_at_zero']=True
        return c,valid,f,s
