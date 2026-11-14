"""Existing Matérn3/2 net-force/derivative prior with176 relative references.

One shared two-force observer: append j after the existing11 states. Reuse
the existing transition and publisher, fixed stationary acceleration variance
and integrated correlation. No trained scene parameters or extra trajectory.
"""
import numpy as np
from layout_reference_observer import LayoutReferenceObserver
from relative_rotor_observer import RelativeRotorForceObserver
from matern_force_observer import transition,RATE


class SmoothLayoutForceObserver(LayoutReferenceObserver):
    state_dim=13
    dynamic_indices=np.array([0,1,2,3,4,5,11,12])

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.consistent_dynamics=True
        transition(33333)

    def _initialize(self,z):
        super()._initialize(z)
        self.P[11:13,11:13]=np.eye(2)*RATE**2*1e6

    def _transition(self,dt):
        A,Qnet=transition(round(dt*1e6));indices=self.dynamic_indices
        F=np.eye(self.state_dim);Q=np.zeros((self.state_dim,self.state_dim))
        F[np.ix_(indices,indices)]=A;Q[np.ix_(indices,indices)]=Qnet
        # Coordinate change a=d+k*u for a fixed previous input over this
        # interval, including the derivative's response. Membership changes
        # retain the inherited net-acceleration-preserving transfer rule.
        F[indices,self.gain_index]=A[:,4:6]@self.u-np.r_[np.zeros(4),self.u,np.zeros(2)]
        return F,Q

    def _input_covariance_map(self,F):
        B=super()._input_covariance_map(F)
        B[11:13]=F[11:13,4:6]
        return B

    def _joint_step(self,t,body,u,*args,**kwargs):
        was_initialized=self.two_point_initialized
        result=RelativeRotorForceObserver.step(self,t,body,u,*args,**kwargs)
        if self.two_point_initialized and not was_initialized:
            # The inherited two-position initialization resets covariance;
            # restore only the newly appended derivative prior, never p/v/d.
            self.P[11:13,11:13]=np.eye(2)*RATE**2*1e6
        result[-1].update(net_force_derivative_state=True,force_prior='Matern32_fixed_integrated_correlation',
            current_input_is_not_a_future_control_command=True)
        return result

    def _force(self,active,acceleration):
        result=super()._force(active,acceleration)
        result.update(future_net_force_model='matern32',force_decay_rate=RATE,
            jerk_px_s3=[0.,0.] if self.x is None else self.x[11:13].tolist(),
            jerk_is_lumped_dynamics_not_measured_rotor_force_rate=True,
            future_net_acceleration_prior='matched_Matern32_net_force_and_derivative',
            integrated_acceleration_correlation_s=.35)
        return result
