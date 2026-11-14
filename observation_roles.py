"""Confirmed rotor presence/motion does not grant a new motor-force input.

Reuse current strict periodic evidence + bipolar reconstruction + two-way
matching. A weak current pixel population may continue an existing target,
but it does not update the motor-frequency bank or manufacture known thrust.
No threshold or per-recording rule is introduced here.
"""
from periodic_track_continuation import PeriodicTrackContinuation


class RoleSeparatedContinuation(PeriodicTrackContinuation):
    def __init__(self,*,prior_consistent=False,allow_period_gap=False):
        super().__init__(appearance_uncertainty=True,structural_motion=True,allow_period_gap=allow_period_gap)
        self.prior_consistent=bool(prior_consistent)
    def prepare(self,frame,front,t):
        info=super().prepare(frame,front,t)
        if info.get('used'):
            # recover() already declared regional_only_motor_frequency_unobserved
            # in its returned metadata. The older continuation copied only the
            # mask/match, losing that qualification before force construction.
            frame['current_support_without_force_update']=True
            frame['continued_observation_preserves_gap_prior']=self.prior_consistent
            frame['regional_only_motor_frequency_unobserved']=True
            frame['observation_qualification']=dict(current_presence_confirmed=True,
                current_displacement_checked=True,current_motor_frequency_qualified=False,
                current_force_bank_update_allowed=False,hidden_rotor_force_owner='joint_unknown_state_d')
            info.update(qualification_roles_separated=True,motor_bank_not_updated=True,
                current_known_force_created=False,current_support_is_real_image_measurement=True)
        return info


def unavailable_force_metadata():
    return dict(source='continued_rotor_support_without_force_update',reason='presence_and_motion_do_not_qualify_new_motor_force',
        frequency_hz=None,relative_f2=None,coherent=False,rotor_bank_used=False,
        learning_input_current=False,current_model_ids=[],current_individual_frequency_hz={},
        observed_components={},input_sigma_proxy=.25,input_uncertainty_calibrated=False,
        mixed_proxy_is_total_thrust=False,mixed_proxy_gain_learning=False,
        hidden_force_owner='joint_unknown_state_d',hidden_frequencies_measured=0,
        metric_thrust_measured=False,mechanical_RPM_measured=False,
        current_support_is_not_force_input=True,motor_frequency_bank_updated=False)
