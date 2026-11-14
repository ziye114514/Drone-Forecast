"""Joint image position/velocity, rotor input and lumped-disturbance observer.

Independent small Kalman model inspired by ETH's disturbance observer. Native
source timestamps determine the transition: a newly measured rotor input never
acts backwards over the previous interval. No body NN, labels, delayed position
residual, metric force, or second trajectory branch is used.
"""
from collections import deque
import numpy as np


class RotorForceObserver:
    state_dim=7
    gain_index=6
    correlation_s = .35

    def __init__(self, default_gain=200.,velocity_sigma=100.,startup_fit=False,consistent_dynamics=False,two_point_init=False):
        self.default_gain = float(default_gain)
        self.velocity_sigma=float(velocity_sigma)
        self.startup_fit=bool(startup_fit);self.startup_samples=[];self.startup_fitted=False
        self.consistent_dynamics=bool(consistent_dynamics)
        self.two_point_init=bool(two_point_init);self.first_position=None;self.two_point_initialized=False
        if self.consistent_dynamics:
            from singer_position_state import transition
            transition(33333)
        self.x = self.P = None
        self.t = self.seen = self.input_t = None
        self.u = np.zeros(2)
        self.observations = 0
        self.inputs = deque()
        self.learned = False
        self.components = {}
        self.input_cov = np.zeros((2, 2))
        self.previous_gain_qualified = False
        np.linalg.solve(np.eye(2),np.ones((2,self.state_dim)))

    def _set_input(self, u, reanchor=False, transfer_delta=None, transfer_cov=None):
        u = np.asarray(u, float)
        if reanchor or transfer_delta is not None:
            # Only evidence entering/leaving the known partition is transferred.
            # Retained motors' real frequency/normal changes still change net a.
            delta = u-self.u if reanchor else np.asarray(transfer_delta, float)
            J = np.eye(self.state_dim); J[4:6,self.gain_index] = -delta
            self.x[4:6] -= self.x[self.gain_index]*delta
            self.P = J@self.P@J.T
            if transfer_cov is not None:
                # Conservative independent-error approximation. The proxy
                # errors are not calibrated or uniquely separable from d.
                self.P[4:6, 4:6] += (self.x[self.gain_index]**2+self.P[self.gain_index,self.gain_index])*transfer_cov
        self.u = u.copy()

    def step(self, t, body, u, quality=0., allow_gain_learning=True, input_meta=None, motion=None,measurement_lag_s=0.):
        lag=float(measurement_lag_s)
        if not 0<=lag<=.05:raise ValueError('Measurement lag must be within0..50ms')
        if lag and motion is not None:raise ValueError('Delayed position with extra flow observations is not implemented')
        z = np.asarray(body['center'], float) if body.get('detected') and not body.get('center_provisional',False) else None
        if self.seen is not None and t-self.seen > .5:
            self.__init__(self.default_gain,self.velocity_sigma,self.startup_fit,self.consistent_dynamics,self.two_point_init)
        reference_shift=body.get('measurement_reference_shift_xy')
        if reference_shift is not None:
            reference_shift=np.asarray(reference_shift,float)
            if reference_shift.shape!=(2,) or not np.isfinite(reference_shift).all():raise ValueError('Finite tracked-reference shift required')
            if self.x is not None:self._rebase_observation_reference(reference_shift)
        if self.x is None:
            if z is None:
                return None, False, self._force(False, None), dict(reason='uninitialized')
            self._initialize(z)
            dt = 0.
        else:
            dt = float(t-self.t)
            if dt <= 0: raise ValueError('Strictly increasing source timestamps required')
            F,Q=self._transition(dt)
            self.x = F@self.x; self.P = F@self.P@F.T+Q
            if input_meta is not None:
                B = self._input_covariance_map(F)
                self.P += (self.x[self.gain_index]**2+self.P[self.gain_index,self.gain_index])*B@self.input_cov@B.T
        self.t = float(t)
        if z is not None:
            self.observations += 1; self.seen = float(t)
            if self.startup_fit and not self.startup_fitted:
                self.startup_samples.append((float(t),z.copy()));self.startup_samples=self.startup_samples[-5:]
        valid = bool(self.observations >= 5 and self.seen is not None and t-self.seen <= .16)
        good = bool(valid and u is not None and quality >= .35 and np.isfinite(u).all())
        new_components = {}; transfer_delta = np.zeros(2); membership_changed = False
        new_cov = np.zeros((2, 2))
        if input_meta is not None:
            new_components = {str(k): np.asarray(v, float) for k, v in
                input_meta.get('observed_components', {}).items()} if good else {}
            if any(v.shape != (2,) or not np.isfinite(v).all() for v in new_components.values()):
                raise ValueError('Finite two-dimensional observed contributions required')
            summed = np.sum(list(new_components.values()), axis=0) if new_components else np.zeros(2)
            if good and (not new_components or not np.allclose(summed, u, rtol=1e-9, atol=1e-9)):
                raise ValueError('Observed components must sum to current input')
            old_keys = set(self.components); new_keys = set(new_components)
            membership_changed = old_keys != new_keys
            for key in new_keys-old_keys: transfer_delta += new_components[key]
            for key in old_keys-new_keys: transfer_delta -= self.components[key]
            sigma = float(input_meta.get('input_sigma_proxy', 0.)) if good else 0.
            if not np.isfinite(sigma) or sigma < 0: raise ValueError('Invalid input uncertainty')
            new_cov = np.eye(2)*sigma*sigma
            if membership_changed: self.inputs.clear()
        while self.inputs and t-self.inputs[0][0] > 3.: self.inputs.popleft()
        history = np.array([v for _, v in self.inputs])
        excitation = float(np.sqrt(np.mean(np.sum((history-history.mean(0))**2, axis=1)))) if len(history) else 0.
        learn = self._gain_learning_allowed(good,allow_gain_learning,len(history),excitation,membership_changed,input_meta)
        innovation = None; gain_changed = False;flow_used=False
        if z is not None:
            H,R,measurement,flow_used=self._measurement(z,body,motion,lag)
            residual=measurement-H@self.x;S=H@self.P@H.T+R
            K=np.linalg.solve(S,H@self.P).T
            innovation = float(residual@np.linalg.solve(S, residual))
            # Do not identify an input gain from constant input or warm-up.
            if not learn: K[self.gain_index] = 0.
            else:self._adjust_gain_update(K,residual,dt,excitation)
            delta = K@residual; self.x += delta
            I = np.eye(self.state_dim)-K@H
            self.P = I@self.P@I.T+K@R@K.T
            self.P = .5*(self.P+self.P.T)
            gain_changed = bool(abs(delta[self.gain_index]) > 1e-10)
            self.learned = self.learned or gain_changed
        startup_applied=False
        if z is not None and self.two_point_init:
            if self.first_position is None:self.first_position=(float(t),z.copy())
            elif self.observations==2 and not self.two_point_initialized:
                span=t-self.first_position[0]
                if 0<span<=.12:
                    # Two-position initialization with a diffuse initial
                    # velocity, instead of the unjustified v=0 +/-100px/s.
                    # Current v is the interval mean plus an uncertain half
                    # acceleration step. Retain its p/v and v/a correlations.
                    self.x[:2]=z;self.x[2:4]=(z-self.first_position[1])/span
                    self.x[4:6]=-self.x[self.gain_index]*self.u
                    C=np.array([[36.,36./span,0.],
                        [36./span,72./span**2+.25*span**2*1e6,.5*span*1e6],
                        [0.,.5*span*1e6,1e6]])
                    gain_variance=self.P[self.gain_index,self.gain_index];self.P=np.zeros((self.state_dim,self.state_dim));self.P[self.gain_index,self.gain_index]=gain_variance
                    for axis in range(2):self.P[np.ix_([axis,axis+2,axis+4],[axis,axis+2,axis+4])]=C
                    J=np.eye(self.state_dim);J[4:6,self.gain_index]=-self.u;self.P=J@self.P@J.T
                    self.two_point_initialized=True
        if self.startup_fit and not self.startup_fitted and len(self.startup_samples)==5:
            times=np.array([r[0]-t for r in self.startup_samples]);positions=np.array([r[1] for r in self.startup_samples])
            if np.ptp(times)<=.2 and np.ptp(times)>=.08:
                X=np.c_[np.ones(5),times,.5*times*times]
                # Diffuse initial position/velocity, same1000px/s² net-force
                # prior and6px observation noise as the existing observer.
                precision=X.T@X/36.+np.diag([1e-12,1e-12,1e-6])
                covariance=np.linalg.inv(precision);fit=covariance@(X.T@positions/36.)
                residual=np.sqrt(np.mean((X@fit-positions)**2))
                if residual<=6.:
                    k=self.x[self.gain_index];gain_variance=self.P[self.gain_index,self.gain_index]
                    self.x[:2]=fit[0];self.x[2:4]=fit[1];self.x[4:6]=fit[2]-k*self.u
                    self.P=np.zeros((self.state_dim,self.state_dim));self.P[self.gain_index,self.gain_index]=gain_variance
                    for axis in range(2):self.P[np.ix_([axis,axis+2,axis+4],[axis,axis+2,axis+4])]=covariance
                    J=np.eye(self.state_dim);J[4:6,self.gain_index]=-self.u;self.P=J@self.P@J.T
                    self.startup_fitted=True;startup_applied=True
        if input_meta is not None:
            # Loss of current evidence transfers it to d immediately; it is
            # never left in the known term as a hidden short-duration motor.
            self._set_input(u if good else np.zeros(2),
                transfer_delta=transfer_delta if membership_changed else None,
                transfer_cov=self.input_cov+new_cov if membership_changed else None)
            self.components = new_components; self.input_cov = new_cov
            self.previous_gain_qualified = bool(good and allow_gain_learning)
            if good:
                self.input_t = float(t)
                if allow_gain_learning: self.inputs.append((float(t), np.asarray(u, float).copy()))
        elif good:
            reacquired = self.input_t is None or t-self.input_t > .12
            self._set_input(u, reanchor=reacquired)
            self.input_t = float(t)
            if allow_gain_learning:self.inputs.append((float(t), np.asarray(u, float).copy()))
        elif self.input_t is None or t-self.input_t > .12:
            # Expiring the rotor evidence must not manufacture an acceleration
            # jump. Keep the estimated net motion, revoke current force credit.
            self._set_input(np.zeros(2), reanchor=True)
        acceleration = self.x[4:6]+self.x[self.gain_index]*self.u
        norm = float(np.linalg.norm(acceleration))
        if norm > 3000.: acceleration = acceleration*(3000./norm)
        c = dict(position=self.x[:2].tolist(), velocity=self.x[2:4].tolist(),
                 acceleration=acceleration.tolist())
        force = self._force(good, acceleration)
        force.update(gain_learned=gain_changed, gain_ever_learned=self.learned,
            gain_sigma=float(np.sqrt(max(0., self.P[self.gain_index,self.gain_index]))),
            frozen_reason=None if learn else 'warmup_or_insufficient_input_excitation',
            excitation_rms=excitation, qualified_input_history=len(self.inputs))
        if input_meta is not None:
            force.update(force_decomposition='currently_observed_plus_lumped_unobserved',
                hidden_rotor_force_owner='disturbance_state_d',
                observed_components={k:v.tolist() for k,v in self.components.items()},
                visibility_partition_changed=membership_changed,
                transferred_input_u=transfer_delta.tolist(),
                transfer_acceleration_px_s2=(self.x[self.gain_index]*transfer_delta).tolist(),
                transfer_covariance_is_conservative=True,
                input_sigma_proxy=float(np.sqrt(new_cov[0, 0])),
                mixed_proxy_is_total_thrust=False)
        state = dict(observations=self.observations, measurement_innovation_chi2=innovation,
            support_reference_shift_xy=None if reference_shift is None else reference_shift.tolist(),
            support_reference_change_is_not_a_force_impulse=reference_shift is not None,
            measurement_lag_s=lag,
            consistent_update_and_rollout=self.consistent_dynamics,
            initial_velocity_from_two_observations=self.two_point_initialized,
            causal_startup_fit_applied=startup_applied,causal_startup_fit_completed=self.startup_fitted,
            native_translation_velocity_observed=flow_used,
            joint_position_velocity_disturbance=True, parallel_trajectory_branch=False,
            calibration_before_valid=False, past_interval_uses_previous_input=True,
            delayed_position_residual_used=False, position_sigma_px=np.sqrt(np.maximum(0.,np.diag(self.P)[:2])).tolist(),
            velocity_sigma_px_s=np.sqrt(np.maximum(0.,np.diag(self.P)[2:4])).tolist())
        return c, valid, force, state

    def _rebase_observation_reference(self,shift):
        self.x[:2]+=shift
        # Same6px absolute-anchor noise prior. The new pixel reference is a
        # position gauge hypothesis; no velocity/force increment is measured.
        self.P[:2,:2]+=np.eye(2)*36.

    def _measurement(self,z,body,motion,lag):
        H=np.zeros((2,self.state_dim));H[:,:2]=np.eye(2)
        if lag:
            H[:,2:4]=-lag*np.eye(2);H[:,4:6]=.5*lag*lag*np.eye(2)
            H[:,self.gain_index]=.5*lag*lag*self.u
        R=np.eye(2)*36.;measurement=z;flow_used=False
        shift=(motion or {}).get('estimate') or {};flow_dt=(motion or {}).get('dt_s')
        if flow_dt is not None and 0<flow_dt<=.12 and shift.get('score',0)>0:
            delta=np.array([shift.get('dx'),shift.get('dy')],float)
            boundary=float(shift.get('search_px',40))
            if np.isfinite(delta).all() and np.max(abs(delta))<boundary:
                H=np.zeros((4,self.state_dim));H[:4,:4]=np.eye(4)
                measurement=np.r_[z,delta/flow_dt]
                variance=4.;cross=variance/flow_dt if body.get('measurement_source')=='filtered_native_translation' else 0.
                R=np.block([[np.eye(2)*36.,np.eye(2)*cross],
                    [np.eye(2)*cross,np.eye(2)*variance/(flow_dt*flow_dt)]])
                flow_used=True
        return H,R,measurement,flow_used

    def _force(self, active, acceleration):
        return dict(active=bool(active), effective_gain_px_s2=self.default_gain if self.x is None else float(self.x[self.gain_index]),
            input_u=self.u.tolist() if active else None,
            disturbance_px_s2=None if self.x is None else self.x[4:6].tolist(),
            acceleration_px_s2=[0.,0.] if acceleration is None else acceleration.tolist(),
            acceleration_correlation_s=self.correlation_s,
            disturbance_estimator='joint_timestamped_position_velocity_disturbance_KF',
            current_input_used_without_waiting_for_current_future_position=bool(active),
            acceleration_source='qualified_rotor_input_plus_joint_unknown_state' if active else 'joint_net_state_without_current_rotor_credit',
            future_net_acceleration_prior='Singer_zero_mean_conditional_expectation',
            future_commands_observed=False, metric_force_measured=False, direction_sign='image_up_prior')

    def _gain_learning_allowed(self,good,allow_gain_learning,history_length,excitation,membership_changed,input_meta):
        learn=bool(good and allow_gain_learning and history_length>=8 and excitation>=.08)
        if input_meta is not None:learn=bool(learn and self.previous_gain_qualified and not membership_changed)
        return learn

    def _adjust_gain_update(self,K,residual,dt,excitation):
        K[self.gain_index]*=-np.expm1(-dt/1.)
        candidate=self.x[self.gain_index]+float(K[self.gain_index]@residual)
        if not 0.<candidate<5000.:K[self.gain_index]=0.

    def _input_covariance_map(self,F):
        B=np.zeros((self.state_dim,2));B[:2]=np.eye(2)*F[0,4];B[2:4]=np.eye(2)*F[2,4]
        if self.consistent_dynamics:B[4:6]=np.eye(2)*(F[4,4]-1.)
        return B

    def _transition(self,dt):
        F = np.eye(self.state_dim);Q = np.zeros((self.state_dim,self.state_dim))
        if self.consistent_dynamics:
            from singer_position_state import transition
            # Apply the SAME conditional net-acceleration prior used by
            # physical_publication. In [p,v,a,k] it is exactly Singer;
            # a=d+k*u is only a coordinate change. Thus loss/reappearance
            # of a measured contribution cannot change the forecast by
            # changing which force owns the baseline. Gravity is not
            # claimed to physically decay; this is an image-motion prior.
            A,Qnet=transition(round(dt*1e6),self.correlation_s);F[:6,:6]=A;Q[:6,:6]=Qnet
            F[:2,self.gain_index]=A[0,4]*self.u;F[2:4,self.gain_index]=A[2,4]*self.u
            F[4:6,self.gain_index]=(A[4,4]-1.)*self.u
        else:
            # Stage162 control: random-walk disturbance update, followed
            # by a Singer future rollout (inconsistent conditional model).
            F[:2, 2:4] = np.eye(2)*dt
            F[:2, 4:6] = np.eye(2)*(.5*dt*dt)
            F[2:4, 4:6] = np.eye(2)*dt
            F[:2,self.gain_index] = .5*dt*dt*self.u
            F[2:4,self.gain_index] = dt*self.u
            block = np.array([[dt**5/20, dt**4/8, dt**3/6],
                              [dt**4/8, dt**3/3, dt**2/2],
                              [dt**3/6, dt**2/2, dt]])*(2*1000.**2/.35)
            for axis in range(2):
                ix = [axis, 2+axis, 4+axis];Q[np.ix_(ix, ix)] = block
        return F,Q

    def _initialize(self,z):
        self.x = np.r_[z, np.zeros(4), self.default_gain]
        self.P = np.diag([10000.]*4+[1000000.]*2+[self.default_gain**2])
        self.P[2:4,2:4]=np.eye(2)*self.velocity_sigma**2
