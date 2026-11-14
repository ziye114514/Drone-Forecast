"""Rebase a derivative's position-consistency history after a shape reanchor.

    Uses a current strong detector anchor and already accepted rigid tracks.
    Only the old consistency coordinates move. Current localization, velocity
    observations, rotor geometry and the shared position filter stay unchanged.
    A large/unexplained shift is NOT silently removed.
"""
import numpy as np
from collections import deque
from live_local_flow import box
from axis_amplitude_bridge import current_axis_qualified


class PositionHistoryGauge:
    def __init__(self,stable_axis=False):self.axes=deque();self.stable_axis=stable_axis
    def step(self,positions,t,body,previous,observation,structure=None):
        if not self.stable_axis:return rebase(positions,t,body,previous,observation)
        s=structure or {}
        qualified=bool(s.get('mode') in ('axis_only','edge_on') and
            (s.get('axis_support') or {}).get('valid') and current_axis_qualified(s))
        if qualified:
            n=np.asarray(s['axis_xy'],float);n=n/np.linalg.norm(n);self.axes.append((t,n))
        while self.axes and t-self.axes[0][0]>.35+1e-8:self.axes.popleft()
        if not qualified or len(self.axes)<6 or self.axes[-1][0]-self.axes[0][0]<.16-1e-8:
            return dict(applied=False,reason='no_current_stable_side_axis_history')
        times=np.array([r[0] for r in self.axes]);normals=np.array([r[1] for r in self.axes])
        spread=float(np.rad2deg(np.arccos(np.clip(abs(normals@n),0.,1.))).max())
        confidence=float((s.get('axis_support') or {}).get('confidence',s.get('geometric_weight',.45)))
        limit=5./max(.45,min(1.,confidence))
        if np.max(np.diff(times))>.08+1e-8 or spread>limit:
            return dict(applied=False,reason='side_axis_gap_or_changed_projection',axis_spread_deg=spread,gate_deg=limit)
        return dict(rebase(positions,t,body,previous,observation),current_axis_stable=True,
            axis_spread_deg=spread,axis_gate_deg=limit,axis_does_not_require_four_circle_template=True)


def rebase(positions,t,body,previous,observation):
    info=dict(applied=False,reason='no_current_shape_reanchor',current_position_unchanged=True,
        current_velocity_observation_unchanged=True,new_position_filters=0)
    if previous is None or not positions or not observation.get('valid'):return info
    stamp,old=previous;dt=t-stamp
    if not 0.<dt<=.08000001 or box(body) is None or box(old) is None:return info
    fresh=body.get('measurement_source') in ('current_NN','current_KLT_from_delayed_NN') and body.get('nn_observed_t',t)!=old.get('nn_observed_t')
    if not fresh or body.get('confidence',0)<.45 or old.get('confidence',0)<.45:return info
    current=np.asarray(body['bbox']);prior=np.asarray(old['bbox'])
    shape_change=float(np.max(abs((current[2:]-current[:2])-(prior[2:]-prior[:2]))))
    if shape_change<6.:return dict(info,reason='no_shape_change_above_position_noise_floor')
    covariance=np.asarray(observation['velocity_covariance'])*dt**2
    if np.linalg.eigvalsh(covariance).max()>9.:
        return dict(info,reason='current_rigid_translation_too_uncertain')
    shift=np.asarray(body['center'])-np.asarray(old['center'])-np.asarray(observation['displacement'])
    if np.linalg.norm(shift)>12.:
        return dict(info,reason='large_reanchor_not_explained_as_small_shape_gauge',shift_px=shift.tolist())
    for i,(s,p) in enumerate(positions):positions[i]=(s,np.asarray(p)+shift)
    return dict(info,applied=True,reason='current_shape_anchor_rebases_old_consistency_coordinates',
        shift_px=shift.tolist(),shape_change_px=shape_change,history_rows=len(positions),
        current_box_and_force_position_unchanged=True,rotation_and_blade_motion_not_relabelled_as_translation=True)
