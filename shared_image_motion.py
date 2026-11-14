"""Reuse already-computed current KLT displacement; no new optical-flow pass.

Uses existing velocity covariance intersection and current-flow derivative.
The shared position KF and its eligibility stay unchanged. Flow-derived net
acceleration is unobserved total force when rotor periodicity is unavailable.
"""
import copy
import numpy as np
from live_local_flow import box,velocity_coefficient
from live_flow_acceleration import FlowAcceleration


class SharedImageMotion:
    def __init__(self,reanchor_velocity=True,same_window_motion=False,curved_motion=False,anchor_gauge=False,weak_motion_support=False,visible_motion_support=False,spatial_weak_motion=False,position_history_rebase=False,contained_reanchor_motion=False,contiguous_motion_history=False,component_acceleration=False):
        derivative=FlowAcceleration
        if curved_motion:
            from curved_flow_motion import CurvedFlowAcceleration
            derivative=CurvedFlowAcceleration
        if component_acceleration:
            from component_flow_acceleration import ComponentFlowAcceleration
            derivative=ComponentFlowAcceleration
        self.previous=None;self.derivative=derivative('same-window' if same_window_motion else 'state')
        self.derivative.contiguous_history=contiguous_motion_history
        self.reanchor_velocity=reanchor_velocity
        self.contained_reanchor_motion=contained_reanchor_motion
        self.anchor_gauge=None
        self.position_history_gauge=None
        if position_history_rebase:
            from position_history_gauge import PositionHistoryGauge
            self.position_history_gauge=PositionHistoryGauge(stable_axis=position_history_rebase=='stable-axis')
        self.weak_support=None
        self.support_box=box
        if visible_motion_support:
            from motion_observability import visible_motion_box
            self.support_box=visible_motion_box
            self.derivative.support_box=visible_motion_box
        if weak_motion_support:
            from motion_observability import WeakSemanticMotion
            self.weak_support=WeakSemanticMotion(allow_spatial=spatial_weak_motion)
        if anchor_gauge:
            from flow_anchor_gauge import FlowAnchorGauge
            self.anchor_gauge=FlowAnchorGauge()

    def step(self,t,coefficient,velocity_covariance,body,tracking,structure=None):
        prior=self.previous;self.previous=(t,copy.deepcopy(body))
        observation=dict(valid=False,reason='no_qualified_current_shared_translation')
        current_box=self.support_box(body)
        old_box=None if prior is None else self.support_box(prior[1]);upgrade=False
        truncated_pair=bool(prior is not None and current_box is not None and old_box is not None and
            (box(body) is None or box(prior[1]) is None))
        weak_info=None;weak_pair=False;motion_body=body
        if self.weak_support is not None:
            weak_info=self.weak_support.observe(t,body,tracking)
            if weak_info['qualified'] and prior is not None:
                current_candidate=box(dict(body,low_confidence=False))
                old_candidate=box(dict(prior[1],low_confidence=False))
                if current_candidate is not None and old_candidate is not None:
                    weak_pair=current_box is None or old_box is None
                    current_box=current_candidate;old_box=old_candidate
                    if weak_pair:motion_body=dict(body,low_confidence=False)
        new_anchor=bool(prior is not None and current_box is not None and body.get('confidence',0)>=.45 and
            body.get('measurement_source') in ('current_NN','current_KLT_from_delayed_NN') and
            body.get('nn_observed_t',t)!=prior[1].get('nn_observed_t'))
        if (self.reanchor_velocity and prior is not None and old_box is None and current_box is not None and
            prior[1].get('low_confidence') and prior[1].get('confidence',0)>=.25 and
            new_anchor):
            old_box=box(dict(prior[1],low_confidence=False));upgrade=old_box is not None
        if (prior is not None and current_box is not None and old_box is not None and
            tracking.get('reason')=='current_image_translation' and tracking.get('points',0)>=8 and
            0.<t-prior[0]<=.08000001):
            dt=t-prior[0];delta=np.asarray(tracking['delta'],float)
            covariance=np.asarray(tracking.get('displacement_covariance_px2',np.eye(2)*4.),float)
            similarity=tracking.get('similarity_motion') or {}
            if similarity.get('valid'):
                delta=np.asarray(similarity['displacement'],float)
                covariance=np.asarray(similarity['displacement_covariance_px2'],float)
                if weak_pair and (weak_info or {}).get('spatial_support_used'):
                    covariance=covariance+np.asarray(similarity['spatial_support']['extra_displacement_covariance_px2'])
            actual=np.asarray(body['center'])-np.asarray(prior[1]['center']);mismatch=delta-actual
            consistency=float(mismatch@np.linalg.solve(covariance+np.eye(2)*72.,mismatch))
            translated=old_box+np.tile(delta,2)
            intersection=np.prod(np.maximum(0.,np.minimum(translated[2:],current_box[2:])-np.maximum(translated[:2],current_box[:2])))
            union=np.prod(translated[2:]-translated[:2])+np.prod(current_box[2:]-current_box[:2])-intersection
            overlap=float(intersection/max(union,1e-9))
            areas=[float(np.prod(translated[2:]-translated[:2])),float(np.prod(current_box[2:]-current_box[:2]))]
            containment=float(intersection/max(min(areas),1e-9));area_ratio=max(areas)/max(min(areas),1e-9)
            # A reliable partial box can grow when more of the same body is
            # visible. Containment of its measured track region is stronger
            # evidence here than IoU, which penalizes the newly exposed area.
            # Do not give this extra exception to an old weak semantic box.
            contained_strong=bool(not prior[1].get('low_confidence') and prior[1].get('confidence',0)>=.45 and
                containment>=.8 and area_ratio<=4.)
            # A new strong semantic anchor can also confirm that a previously
            # weak tracked patch belonged to this body. Its displacement needs
            # stricter *current* rigid support; the old partial box is never
            # promoted to a full-body localization or rotor observation.
            contained_patch=bool(self.contained_reanchor_motion and upgrade and
                containment>=.8 and area_ratio<=4. and tracking.get('points',0)>=12 and
                similarity.get('valid') and similarity.get('inliers',0)>=12 and
                similarity.get('inlier_fraction',0)>=.75 and
                np.linalg.eigvalsh(covariance).max()<=9.)
            anchor_compatible=bool(self.reanchor_velocity and new_anchor and (overlap>=.5 or contained_strong or contained_patch) and
                np.linalg.eigvalsh(covariance).max()<=36.)
            # A newly strong NN anchor corrects the position gauge. Its center
            # shift is not a velocity measurement. The already measured tracks
            # can still constrain velocity if both current boxes overlap.
            truncated_agreement=not truncated_pair or (similarity.get('valid') and similarity.get('inliers',0)>=12 and
                similarity.get('inlier_fraction',0)>=.75 and np.linalg.eigvalsh(covariance).max()<=9.)
            weak_agreement=(not weak_pair or (overlap>=.5 and consistency<=5.991464547107979)) and truncated_agreement
            if weak_agreement and ((not upgrade and consistency<=5.991464547107979) or anchor_compatible):
                observation=dict(valid=True,reason='reused_current_bidirectional_translation',
                    velocity=(delta/dt).tolist(),velocity_covariance=(covariance/(dt*dt)).tolist(),
                    displacement=delta.tolist(),midpoint_t=.5*(t+prior[0]),box_displacement_chi2=consistency,
                    original_tracks_reused=True,not_independent_of_body_localization=True,
                    covariance_source='observed_full_KLT_scatter' if 'displacement_covariance_px2' in tracking else 'legacy2px_proxy_for_saved_records_only')
                if similarity.get('valid'):observation.update(covariance_source='full_similarity_residual_scatter_plus_localization_floor',
                    current_similarity_reused=True,evaluation_center_source=similarity.get('evaluation_center_source'))
                if anchor_compatible and (upgrade or consistency>5.991464547107979):observation.update(
                    reason='current_strong_NN_anchor_upgrade_with_current_tracks' if upgrade else 'current_strong_NN_visibility_reanchor_with_current_tracks',
                    anchor_upgrade=upgrade,visibility_reanchor=not upgrade,current_projected_box_IoU=overlap,
                    current_projected_box_containment=containment,current_box_area_ratio=area_ratio,
                    position_gauge_correction_not_treated_as_velocity=mismatch.tolist(),
                    previous_weak_box_not_promoted_to_strong_detection=True)
                if contained_patch:observation.update(contained_weak_patch_motion=True,
                    patch_confirmed_by_current_strong_body=True,partial_to_full_center_jump_not_used_for_velocity=True,
                    derivative_eligible=False,
                    derivative_exclusion_reason='tracked_partial_reference_changes_to_full_body_anchor')
        gauge_info=None
        if self.anchor_gauge is not None:observation,gauge_info=self.anchor_gauge.step(t,body,tracking,observation)
        base_acceleration=np.zeros(2) if coefficient is None else np.asarray(coefficient['acceleration'])
        value,velocity_info=velocity_coefficient(coefficient,velocity_covariance,observation,t)
        history_gauge=None
        if self.position_history_gauge is not None:
            history_gauge=self.position_history_gauge.step(self.derivative.positions,t,motion_body,prior,observation,structure)
        value,acceleration_info=self.derivative.step(t,value,motion_body,observation,'position-consistent')
        correction=np.zeros(2)
        if velocity_info.get('applied') and acceleration_info.get('applied'):
            # The first CI used its prior acceleration to align midpoint flow.
            # Replace only that half-step term using the newly qualified slope;
            # do not fuse the same flow observation twice.
            weight=velocity_info['baseline_weight'];P=np.asarray(velocity_covariance);R=np.asarray(observation['velocity_covariance'])
            invP,invR=np.linalg.inv(P),np.linalg.inv(R);C=np.linalg.inv(weight*invP+(1-weight)*invR)
            correction=C@((1-weight)*invR)@((t-observation['midpoint_t'])*(np.asarray(value['acceleration'])-base_acceleration))
            value=dict(value,velocity=(np.asarray(value['velocity'])+correction).tolist())
        return value,dict(observation=observation,velocity=velocity_info,acceleration=acceleration_info,
            anchor_gauge=gauge_info,
            position_history_gauge=history_gauge,
            contiguous_history_restart=self.derivative.last_restart,
            weak_semantic_motion=weak_info,
            truncated_body_motion_considered=truncated_pair,
            midpoint_alignment_correction_px_s=correction.tolist(),new_flow_passes=0,new_position_filters=0,
            position_and_validity_unchanged=True,acceleration_is_not_measured_rotor_thrust=True)
