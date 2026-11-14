"""Refit one seeded activity core from its current native bipolar outline.

The dense count4 core localizes activity, not necessarily the disk boundary.
Reuse147's bounded original-contour fitter and occupancy convention. A parent
with multiple cores or multiple concavities still belongs to the pair splitter.
No new periodic votes, missing hubs, cut chords or historical pixels are made.
"""
import cv2
import numpy as np
from batched_arc_fit import fit_arc_exact
from curvature_visibility import masks, contours
from research.stage147.curvature147 import notches


def recover_isolated_rims(on, off, existing, offset):
    _, outline = masks(on, off)
    offset = np.asarray(offset, float)
    rotors = [dict(r) for r in existing]
    audit = []
    trials = 0
    for parent in contours(outline)[:3]:
        ids = [i for i, r in enumerate(rotors)
               if r.get('current_observation') and not r.get('fov_clipped')
               and (r.get('mode') == 'ellipse' or r.get('arc_center_qualified'))
               and cv2.pointPolygonTest(parent, (float(r['cx']-offset[0]),
                                                float(r['cy']-offset[1])), False) >= 0]
        record = dict(accepted=False, existing_cores=len(ids))
        audit.append(record)
        if len(ids) != 1:
            record['reason'] = 'need_one_current_core_not_zero_or_a_merged_pair'
            continue
        if len(notches(parent)) > 1:
            record['reason'] = 'concave_parent_reserved_for_pair_splitter'
            continue
        points = parent.reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(parent)
        if x <= 0 or y <= 0 or x+w >= outline.shape[1] or y+h >= outline.shape[0]:
            record['reason'] = 'outline_touches_ROI_boundary'
            continue
        old = rotors[ids[0]]
        part = np.zeros_like(outline)
        cv2.drawContours(part, [parent], -1, 1, -1)
        native = (outline > 0) & (part > 0)
        old_area = np.zeros_like(outline)
        cv2.ellipse(old_area, tuple(np.rint([old['cx'], old['cy']]-offset).astype(int)),
                    tuple(np.maximum(1, np.rint([old['major']/2, old['minor']/2]).astype(int))),
                    old['angle'], 0, 360, 1, -1)
        old_covered = int(np.sum(native & (old_area > 0)))
        fraction = old_covered / max(1, int(native.sum()))
        record['old_outline_explained'] = fraction
        if fraction >= .60:
            record['reason'] = 'current_core_already_explains_outline'
            continue
        trials += 1
        fit = fit_arc_exact(points, float(np.ptp(points, axis=0).max()))
        if fit is None:
            record['reason'] = 'original_contour_fit_failed'
            continue
        q = fit['parameters']
        if q[3]/q[2] <= .16:
            record['reason'] = 'flat_outline_needs_axis_branch'
            continue
        if fit['angle_sigma'] > 15:
            record['reason'] = 'outline_axis_uncertain'
            continue
        area = np.zeros_like(outline)
        cv2.ellipse(area, tuple(np.rint(q[:2]).astype(int)),
                    tuple(np.maximum(1, np.rint(q[2:4]).astype(int))),
                    float(np.rad2deg(q[4])), 0, 360, 1, -1)
        explained = int(np.sum(native & (area > 0)))
        occupancy = explained / max(1, int(area.sum()))
        delta = np.array([old['cx'], old['cy']])-offset-q[:2]
        co, si = np.cos(q[4]), np.sin(q[4])
        normalized_core = ((delta[0]*co+delta[1]*si)/q[2])**2 + ((-delta[0]*si+delta[1]*co)/q[3])**2
        record.update(new_occupancy=occupancy, explained_native_area=explained,
                      normalized_core_distance=float(normalized_core))
        if occupancy < .60 or explained <= old_covered or normalized_core > 1:
            record['reason'] = 'no_supported_expansion_containing_current_core'
            continue
        center = q[:2]+offset
        extent = np.sqrt(np.array([[co*co, si*si], [si*si, co*co]]) @ (q[2:4]**2))
        if np.any(center-extent < 0) or np.any(center+extent >= [1280, 720]):
            record['reason'] = 'sensor_clipped_outline'
            continue
        # Preserve the seed role explicitly. The surrounding occupancy pixels
        # acquire geometric support only; they never acquire frequency votes.
        rotors[ids[0]] = dict(old, cx=float(center[0]), cy=float(center[1]),
            major=float(2*q[2]), minor=float(2*q[3]), ratio=float(q[3]/q[2]),
            angle=float(np.rad2deg(q[4])), mode='arc', inferred_shape=True,
            arc_center_qualified=True, curvature_validated=False,
            center_sigma_px=fit['sigma'], axis_angle_sigma_deg=fit['angle_sigma'],
            center_covariance_px2=(np.eye(2)*fit['sigma']**2).tolist(),
            normal=(np.array([-si,co])*np.sqrt(max(0.,1-(q[3]/q[2])**2))).tolist(),
            fit_condition=fit['condition'], fit_residual_px=fit['rms'],
            visible_arc_fraction=fit['coverage'], support_fraction=fit['support_fraction'],
            geometry_quality=.6, uncertainty_source='current_original_bipolar_outline_local_fit_approximation',
            uncertainty_approximate=True, native_rim_refit=True,
            activity_core_center_xy=[old['cx'], old['cy']],
            activity_core_major_px=old['major'],
            rim_source='current_seeded_native_bipolar_original_contour',
            outline_is_periodicity_measurement=False, no_cut_chord=True)
        record.update(accepted=True, reason='supported_isolated_outline_refit',
                      old_major=old['major'], new_major=float(2*q[2]),
                      old_center=[old['cx'], old['cy']], new_center=center.tolist())
    return rotors, dict(parents=audit, accepted=sum(a['accepted'] for a in audit),
                        arc_fit_trials=trials, current_rotor_count_changed=False,
                        outline_is_periodicity_measurement=False)
