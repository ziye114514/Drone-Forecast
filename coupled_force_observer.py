"""One [position,velocity,unknown acceleration] state, conditional event input.

Reuses the existing LumpedDisturbance observer and slow effective-gain learner.
Missing event thrust becomes part of the unknown force rather than zero total
acceleration. This is not a Newton-thrust or wind/mass identification result.
"""
import numpy as np
from lumped_disturbance import LumpedDisturbance
from fast_force_response import FastForceResponse


class CoupledForceObserver:
    def __init__(self,default_gain=200.):
        self.observer=LumpedDisturbance();self.learner=FastForceResponse(default_gain=default_gain)
        self.seen=None;self.observations=0

    @property
    def gain(self):return self.learner.gain

    def step(self,t,position,u,quality=0.,measurement_scale=1.):
        if self.seen is not None and t-self.seen>.5:
            gain=self.gain;self.__init__(gain)
        if position is not None:self.seen=t;self.observations+=1
        good=bool(u is not None and np.isfinite(u).all() and quality>=.35)
        known=None if not good else self.gain*np.asarray(u,float)
        self.observer.sigma=6.*float(measurement_scale)
        coefficient,observed=self.observer.step(t,position,known)
        before=self.gain
        _,learned=self.learner.step(t,coefficient,position,u if good else None,quality)
        if good and self.observer.x is not None:
            # Gain updates change the decomposition only. Keep the total
            # acceleration continuous, including the next propagated input.
            delta=self.gain-before;self.observer.x[4:6]-=delta*np.asarray(u)
            self.observer.previous_u=self.gain*np.asarray(u)
        predictions={}
        if coefficient is not None:
            p,v,a=[np.asarray(coefficient[k]) for k in ['position','velocity','acceleration']]
            tau=.35
            predictions={str(h):(p+h*v+tau*(h+tau*np.expm1(-h/tau))*a).tolist() for h in (.1,.2,.4)}
        # Retain the baseline publication rule; do not remove bad forecast slots
        # by introducing the older prototype's stricter6-observation gate.
        valid=bool(coefficient is not None and self.observations>=5 and self.seen is not None and t-self.seen<=.16)
        diag=dict(learned,active=bool(good and coefficient is not None),
            acceleration_px_s2=None if coefficient is None else coefficient['acceleration'],
            disturbance_px_s2=observed['disturbance_acceleration'],state_observer=observed,
            disturbance_includes_unobserved_thrust=not good,one_position_velocity_disturbance_filter=True,
            acceleration_source='current_event_input_plus_joint_unknown_force' if good else 'joint_total_unknown_force',
            rollout='Singer_conditional_net_acceleration_tau0.35s',physical_force_measured=False,
            validity_rule='retained5current_positions_and0.16s_age; no extra readyness exclusion')
        return coefficient,predictions,valid,diag
