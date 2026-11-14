"""During a measurement gap, propagate the same conditional net-force prior
as the publisher. Keep the existing observed-frame update recipe unchanged.

This is not a second track, a learned future command, or a new force law.
The inherited observed/unknown partition and relative-state cross covariance
are preserved. Long gaps still expire via the existing seen/valid policy.
"""
from relative_rotor_observer import RelativeRotorForceObserver
from singer_position_state import transition


class CoherentGapObserver(RelativeRotorForceObserver):
    gap_correlation_s=None
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        transition(33333)

    def step(self,t,body,u,*args,**kwargs):
        missing=not body.get('detected') or body.get('center_provisional',False)
        continued=bool(body.get('continued_observation_preserves_gap_prior'))
        conditional_prior=bool(missing or continued)
        previous=self.consistent_dynamics
        previous_tau=self.correlation_s
        # No current position innovation can change the state in this case.
        # Splitting that interval into empty ticks must not silently change
        # the mean from the already-published Singer conditional expectation.
        self.consistent_dynamics=bool(previous or conditional_prior)
        if conditional_prior and self.gap_correlation_s is not None:self.correlation_s=self.gap_correlation_s
        try:
            c,valid,f,s=super().step(t,body,u,*args,**kwargs)
        finally:
            self.consistent_dynamics=previous
            self.correlation_s=previous_tau
        s.update(coherent_measurement_gap_prediction=True,
            no_current_position_evidence=bool(missing),
            gap_transition_uses_published_net_force_prior=bool(missing),
            observed_frame_update_recipe_changed=False)
        if missing:s['gap_correlation_s']=f.get('acceleration_correlation_s')
        if continued:
            s.update(current_observation_uses_existing_gap_prior=True,
                observation_qualification_does_not_switch_transition=True,
                weak_observation_is_assimilated=True,
                observed_frame_update_recipe_changed=True)
        return c,valid,f,s


class ShortGapObserver(CoherentGapObserver):
    """Existing0.1s short-memory prior only while position evidence is absent.

    A conservative observability policy, not an identified force time constant.
    Its covariance uses the same stationary acceleration variance, hence more
    process diffusion rather than treating a prediction as a new measurement.
    Current visible inputs still use the inherited observed/unknown partition.
    """
    gap_correlation_s=.1
