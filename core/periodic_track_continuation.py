"""Separate whole-image birth support from an existing rotor track's evidence.

Cold entry retains the original12 strict pixels. Continuing a confirmed
track requires at least one CURRENT strict period pixel in the same current
bipolar component plus the existing bidirectional native-image match. Every
strict pixel passed the unchanged native discovery/late validation tests.
This refreshes identity, never an individual-motor RPM/thrust measurement.
"""
import numpy as np
from rotor_spatial_bridge import recover


class PeriodicTrackContinuation:
    def __init__(self,*,appearance_uncertainty=False,structural_motion=False,reference_uncertainty=False,allow_period_gap=False):
        self.last_period_t=None;self.last_info=None
        self.appearance_uncertainty=appearance_uncertainty;self.structural_motion=structural_motion
        self.reference_uncertainty=reference_uncertainty
        self.allow_period_gap=bool(allow_period_gap)

    def prepare(self,frame,front,t):
        native,diag=frame['native_precomputed'];strict=frame['period_mask']>0
        regional=bool(diag.get('regional_seed_groups',0)>0 and diag.get('regional_seed_pixels',0)>=3)
        strong=bool(native.sum()>=12 and (frame['counts']['period_pixels']>=12 or regional))
        info=dict(used=False,raw_cold_entry_qualified=strong,current_strict_period_refresh=False,
            periodic_pixel_test_thresholds_unchanged=True,current_RPM_or_thrust_created=False)
        if strong:
            self.last_info=dict(info,reason='original_cold_entry_currently_qualified');return self.last_info
        # Do not invent new periodic seeds. Existential current strict evidence
        # is necessary even when enough current density pixels are available.
        if not np.any(strict) and not self.allow_period_gap:
            self.last_info=dict(info,reason='no_current_strict_period_evidence');return self.last_info
        raw=(frame['on'],frame['off'],dict(native_precomputed=frame['native_precomputed']),frame['density'],frame['density_local'],dict(accepted_pixels=frame['counts']['period_pixels'],mask=frame['period_mask']))
        candidate,proof=recover(raw,front,t,self.last_period_t,include_displacement=True)
        if not proof.get('used'):
            self.last_info=dict(info,reason=proof.get('reason'),continuation_proof=proof);return self.last_info
        support=candidate[2]['native_precomputed'][0];current=int(np.count_nonzero(support&strict))
        if current<1 and (not self.allow_period_gap or np.any(strict)):
            self.last_info=dict(info,reason='current_period_pixels_not_on_matched_target',continuation_proof=proof);return self.last_info
        verified=dict(candidate[2]['verified_native_displacement'])
        structure_motion=None
        if self.structural_motion:
            from native_plane_motion import estimate_plane_motion
            structural,structure_motion=estimate_plane_motion(front.previous,front.previous_mask,(frame['on'],frame['off']),support,front.center,propagate_reference_uncertainty=self.reference_uncertainty)
            if structural is not None and structure_motion.get('qualified'):
                structural=dict(structural,current_displacement_qualified=True,
                    current_geometric_consensus_not_single_blob_translation=True)
                verified['estimate']=structural
        if self.appearance_uncertainty:
            from polarity_motion_quality import qualify
            shift=verified['estimate'];search=candidate[2]['verified_native_displacement']['estimate'];options=dict(native_matching=front.fast_geometry,subpixel=front.subpixel_motion,
                max_shift=search['search_px'],predicted_shift=search['search_center_px'],coarse_matching=front.coarse_motion,
                interior_border=front.border_motion,centered_matching=front.centered_motion)
            verified['estimate']=qualify(front.previous,front.previous_mask,(frame['on'],frame['off']),support,shift,**options)
        frame['native_precomputed']=candidate[2]['native_precomputed']
        frame['current_spatial_track_qualified']=True
        frame['verified_native_displacement']=verified
        frame['current_spatial_track']=dict(reason='current_strict_period_on_bidirectionally_matched_native_component',
            current_period_identity_confirmed=bool(current),current_individually_confirmed_period_pixels=current,
            current_individual_motor_frequency_measured=False,previous_identity_age_s=t-self.last_period_t)
        self.last_info=dict(info,used=True,current_strict_period_refresh=bool(current),current_strict_pixels_in_track=current,
            reason='strict_current_period_and_bidirectional_native_match' if current else 'bounded_period_gap_with_current_bipolar_match',continuation_proof=proof,
            inherited_position_match_reused_without_a_second_search=True,
            ONOFF_appearance_uncertainty=verified['estimate'].get('polarity_motion_quality'),
            current_shared_rotor_motion=structure_motion,
            joint_displacement_not_averaged_with_polarity_estimates=True)
        if not current:
            frame['current_spatial_track'].update(reason='current_image_with_recent_but_not_refreshed_rotor_identity',
                current_period_measured=False,period_identity_clock_refreshed=False)
            self.last_info.update(period_gap_identity_age_s=t-self.last_period_t,
                period_identity_clock_refreshed=False,additional_native_pixels_are_not_new_period_measurements=True)
        return self.last_info

    def observe(self,t,body):
        info=self.last_info or {}
        if body.get('detected') and not body.get('center_provisional') and (info.get('raw_cold_entry_qualified') or info.get('current_strict_period_refresh')):
            self.last_period_t=t
