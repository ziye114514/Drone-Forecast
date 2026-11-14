"""Thresholded ON/OFF observations and explicit four/three/two-rotor hypotheses.

Uses the existing merge-safe, line-first contour frontend (110/124), SDK pixel
period estimator (79), and geometry utilities. Completed hubs are never passed
back to the image detector or counted as observed rotors. All geometry below is
weak-perspective; two points alone are deliberately not treated as a 3-D pose.
"""
from collections import deque
from pathlib import Path
from itertools import permutations
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'research/stage124'),
                str(ROOT / 'research/stage79')]
from occupancy124 import SupportFrontend
import native79

PARAMS = dict(step_us=20000, threshold_units='blurred min(N_ON,N_OFF) per 20 ms',
              threshold_default=4., thin_fallback_threshold=2., split_threshold_factor=2.,
              three_max_ratio_spread=1.6, adaptive_percentile=70., model_expiry_s=.3,
              side_max_ratio=.16, side_axis_tolerance_deg=15.,
              frequency_min_pixels=8, frequency_mode_fraction=.35,
              minimum_assumed_blades=2, missing_rate_max_age_s=.12,
              missing_rate_fallback='equal speeds of hidden and visible rotors, explicitly hypothetical',
              geometry_reuse='110 merge-safe components, 61 line-first geometry, 124 mask provenance',
              frequency_reuse='79 installed SDK FrequencyAlgorithm strict7; original events before spatial suppression',
              one_or_zero='allowed observation failure, baseline forecast retained',
              physical_units='relative squared blade-passage-frequency proxy; not calibrated Newtons/RPM')


def centres(rotors):
    return np.array([[r['cx'], r['cy']] for r in rotors], float).reshape(-1, 2)


def unit(x):
    x = np.asarray(x, float)
    return x / max(np.linalg.norm(x), 1e-12)


def metric(rotors):
    """Inverse projected-circle metric, up to unknown common scale."""
    matrices = []
    for r in rotors:
        angle = np.deg2rad(r['angle']); c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        matrices.append(rotation @ np.diag([1., 1. / max(r['ratio'], .08)**2]) @ rotation.T)
    return np.mean(matrices, axis=0)


def ellipse_tensor(r):
    angle = np.deg2rad(r['angle']); c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return rotation @ np.diag([(r['major']/2)**2, (r['minor']/2)**2]) @ rotation.T


def split_provenance(parents, children):
    """A measured 3 -> 4 transition must split exactly one existing component.

    This never produces a fourth disk by clustering to k=4: all four children
    have to exist as separate contours at the higher threshold.
    """
    result = dict(valid=False, reason='need_three_parents_four_measured_children')
    if len(parents) != 3 or len(children) != 4: return result
    if any(r.get('mode') != 'ellipse' or r.get('inferred_shape') for r in children): return result
    distance = np.linalg.norm(centres(children)[:, None]-centres(parents)[None, :], axis=2)
    scales = np.array([max(r['major'], 1.) for r in parents])
    owners = np.argmin(distance/scales, axis=1)
    counts = np.bincount(owners, minlength=3)
    result.update(child_parent_indices=owners.tolist(), child_counts=counts.tolist())
    if sorted(counts.tolist()) != [1, 1, 2] or np.any(distance[np.arange(4), owners]/scales[owners] > .65):
        result['reason'] = 'higher_threshold_not_a_local_parent_split'; return result
    merged = int(np.argmax(counts)); indices = np.flatnonzero(owners == merged)
    parent_area = max(parents[merged]['major']*parents[merged]['minor'], 1.)
    fractions = [children[i]['major']*children[i]['minor']/parent_area for i in indices]
    # Areas from different thresholds are not comparable: the high threshold
    # retains inner activity cores. Use independently qualified contour sizes,
    # retaining area shrinkage as a diagnostic, not a full-rim assertion.
    all_sizes = np.array([r['major'] for r in children])
    if min(children[i]['major'] for i in indices) < max(10., .5*np.median(all_sizes)):
        result['reason'] = 'higher_threshold_child_too_small_relative_to_other_children'; return result
    result.update(valid=True, reason='independent_higher_threshold_children', split_parent=merged,
                  child_area_fractions=fractions, new_rotor_is_inferred=False,
                  contours_are_thresholded_activity_not_proven_full_physical_rims=True)
    return result


def three_affine_hypotheses(rotors):
    """Infer a missing rectangular-layout hub using ellipse-whitened angles.

    Three unlabelled points alone admit three parallelograms. Coplanar equal
    circular disks supply an affine metric, which can resolve the right corner.
    This is not an exact projective reconstruction and may remain ambiguous.
    """
    p = centres(rotors)
    if len(p) != 3 or any(r.get('mode') != 'ellipse' or r.get('inferred_shape')
                           or r['ratio'] < .16 for r in rotors):
        return dict(valid=False, reason='three_need_three_nonflat_measured_ellipses', hypotheses=[])
    singular = np.linalg.svd(p - p.mean(0), compute_uv=False)
    if singular[1] / max(singular[0], 1e-9) < .08:
        return dict(valid=False, reason='three_near_collinear', hypotheses=[])
    G = metric(rotors)
    sizes = np.array([r['major'] for r in rotors])
    if sizes.max() / sizes.min() > 1.6:
        return dict(valid=False, reason='unequal_projected_disk_sizes', hypotheses=[])
    ratios = np.array([r['ratio'] for r in rotors])
    if ratios.max()/ratios.min() > PARAMS['three_max_ratio_spread']:
        return dict(valid=False, reason='three_shapes_not_one_affine_plane_possible_merged_disk', hypotheses=[])
    hypotheses = []
    for corner in range(3):
        j, k = [i for i in range(3) if i != corner]
        a, b = p[j] - p[corner], p[k] - p[corner]
        cosine = abs(float(a @ G @ b)) / max(np.sqrt((a @ G @ a) * (b @ G @ b)), 1e-9)
        inferred = p[j] + p[k] - p[corner]
        hypotheses.append(dict(corner=corner, orthogonality_error=float(cosine),
                               inferred_center=inferred.tolist()))
    hypotheses.sort(key=lambda q: q['orthogonality_error'])
    best = hypotheses[0]
    margin = hypotheses[1]['orthogonality_error'] - best['orthogonality_error']
    valid = best['orthogonality_error'] <= .25 and margin >= .20
    return dict(valid=valid, reason='weak_perspective_rectangle' if valid else 'three_assignment_ambiguous',
                inferred_center=best['inferred_center'], margin=float(margin),
                hypotheses=hypotheses, inferred_is_observation=False)


def two_axis(rotors):
    p = centres(rotors)
    if len(p) != 2:
        return dict(valid=False, reason='need_two')
    delta = p[1] - p[0]; distance = float(np.linalg.norm(delta))
    direction = unit(delta)
    if distance < max(12., .35 * np.median([r['major'] for r in rotors])):
        return dict(valid=False, reason='two_centres_too_close')
    alignment = [abs(float(direction @ np.array([np.cos(np.deg2rad(r['angle'])),
                                                np.sin(np.deg2rad(r['angle']))]))) for r in rotors]
    flat = all(r['ratio'] <= PARAMS['side_max_ratio'] or r.get('mode') == 'line' for r in rotors)
    valid = flat and min(alignment) >= np.cos(np.deg2rad(PARAMS['side_axis_tolerance_deg']))
    normal = np.array([-direction[1], direction[0]])
    return dict(valid=bool(valid), reason='two_flat_aligned_components' if valid else 'two_is_not_proof_of_side_view',
                center=p.mean(0).tolist(), tangent=direction.tolist(),
                normal_candidates=[normal.tolist(), (-normal).tolist()],
                alignment=[float(a) for a in alignment], separation_px=distance,
                full_3d_plane_observed=False, physical_motor_identity_resolved=False)


def plane_axis(rotors):
    """Unsigned weak-perspective normal, with an explicit image-up sign prior."""
    normals = []
    for r in rotors:
        a = np.deg2rad(r['angle'])
        n = np.array([-np.sin(a), np.cos(a)])
        if n[1] > 0: n = -n
        normals.append(n * np.sqrt(max(0., 1. - r['ratio']**2)))
    vector = np.mean(normals, axis=0)
    agreement = np.linalg.norm(np.mean([unit(v) for v in normals], axis=0))
    return vector, float(agreement)


class ThresholdFrontend(SupportFrontend):
    def __init__(self, w, h, threshold=2., adaptive=False):
        super().__init__(w, h, 'count_roi')
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError('The ON/OFF count threshold must be finite and positive')
        self.threshold146 = float(threshold)
        self.adaptive146 = bool(adaptive)
        self.capture = True
        self.mask146 = np.zeros((h, w), np.uint8)

    def support_mask(self, blurred, threshold):
        selected = blurred
        if self.current_roi is not None:
            x1, y1, x2, y2 = np.rint(self.current_roi).astype(int)
            selected = blurred[max(0, y1):min(self.h, y2), max(0, x1):min(self.w, x2)]
        positive = selected[selected > 0]
        chosen = self.threshold146
        if self.adaptive146 and len(positive):
            chosen = max(chosen, float(np.percentile(positive, PARAMS['adaptive_percentile'])))
        self.support_threshold = chosen
        self.current_density = blurred.copy()
        self.current_raw_mask = np.uint8(blurred >= chosen) * 255
        self.mask146 = self.current_raw_mask.copy()
        return self.current_raw_mask


class BranchGeometry:
    def __init__(self):
        self.hubs = None; self.body = None; self.pose_t = None
        self.full_t = None; self.last_rates = {}; self.shapes = None

    def step(self, t, body, rotors):
        n = len(rotors); p = centres(rotors)
        if self.pose_t is not None and t - self.pose_t > PARAMS['model_expiry_s']:
            self.__init__()
        out = dict(observed_count=n, state='insufficient', model_centers=None,
                   observed_ids=list(range(n)), inferred_ids=[], confidence=0.,
                   axis_xy=None, sign_prior=None, reason='zero_or_one_observation',
                   observed_rotors=rotors, full_3d_pose=False)
        if n not in (2, 3, 4): return out
        if n == 2:
            line = two_axis(rotors); out['two_axis'] = line
            if not line['valid']:
                out.update(state='two_unresolved', reason=line['reason']); return out
            normal = np.array(line['normal_candidates'][0])
            if normal[1] > 0: normal = -normal
            out.update(state='two_side', reason=line['reason'], axis_xy=normal.tolist(),
                       sign_prior='image_up_assumption', confidence=.35,
                       model_centers=p.tolist(), observed_ids=[0, 1],
                       total_thrust_not_identifiable_from_two=True)
            return out
        if any(r.get('inferred_shape') or r.get('fov_clipped') or r.get('mode') != 'ellipse' for r in rotors):
            out.update(state=f'{n}_unresolved', reason='not_independent_complete_disks'); return out
        normal, agreement = plane_axis(rotors)
        if agreement < .85:
            out.update(state=f'{n}_unresolved', reason='ellipse_axes_disagree'); return out
        if n == 4:
            order = np.argsort(np.arctan2(*(p - p.mean(0))[:, ::-1].T))
            ordered = p[order]
            diagonal_error = np.linalg.norm(ordered[0] + ordered[2] - ordered[1] - ordered[3])
            singular = np.linalg.svd(p-p.mean(0), compute_uv=False)
            # Perspective does not preserve parallelograms. Four distinct hubs
            # may legitimately form a trapezoid (the reviewed C frame does).
            # Reject collapsed/concave constellations rather than imposing an
            # affine parallelogram on the image of a 3-D rectangle.
            if len(cv2.convexHull(p.astype(np.float32))) != 4 or singular[1]/max(singular[0], 1e-9) < .08:
                out.update(state='four_unresolved', reason='four_centres_collapsed_or_nonconvex'); return out
            a, b, c, d = np.c_[ordered, np.ones(4)]
            center_h = np.cross(np.cross(a, c), np.cross(b, d))
            projective_center = (center_h[:2]/center_h[2]).tolist()
            ids = np.arange(4)
            if self.hubs is not None and body.get('detected') and self.body is not None:
                expected = self.hubs + np.array(body['center']) - self.body
                alternatives = [(np.linalg.norm(p - expected[list(ids)], axis=1).mean(), ids)
                                for ids in permutations(range(4))]
                _, ids = min(alternatives, key=lambda q: q[0]); ids = np.array(ids)
                hubs = np.zeros((4, 2)); hubs[ids] = p
            else: hubs = p.copy()
            out.update(state='four', reason='four_measured_convex_coplanar_hub_hypothesis', observed_ids=ids.tolist(),
                       model_centers=hubs.tolist(), confidence=.8, axis_xy=normal.tolist(),
                       sign_prior='image_up_assumption', diagonal_affine_error_px=float(diagonal_error),
                       projected_diagonal_intersection=projective_center,
                       image_ellipse_centres_approximate_physical_hubs=True)
            self.full_t = t
            shapes = np.zeros((4, 2, 2))
            for index, rotor_id in enumerate(ids): shapes[rotor_id] = ellipse_tensor(rotors[index])
        else:
            completed = None
            if self.hubs is not None and self.pose_t is not None and body.get('detected') and self.body is not None:
                expected = self.hubs + np.array(body['center']) - self.body
                options = [(float(np.mean(np.linalg.norm(p - expected[list(ids)], axis=1)**2)), ids)
                           for ids in permutations(range(4), 3)]
                options.sort(key=lambda q: q[0]); cost, ids = options[0]
                A = cv2.getAffineTransform(expected[list(ids)].astype(np.float32), p.astype(np.float32))
                singular = np.linalg.svd(A[:, :2], compute_uv=False)
                good = (np.sqrt(cost) <= max(8., .6 * np.median([r['major'] for r in rotors]))
                        and np.linalg.det(A[:, :2]) > 0 and singular.min() > .6 and singular.max() < 1.6
                        and options[1][0] - cost > 4.)
                if good and self.shapes is not None:
                    predicted_shapes = np.array([A[:, :2] @ shape @ A[:, :2].T for shape in self.shapes])
                    axis_ratios = []
                    for observed, rotor_id in zip(rotors, ids):
                        predicted_axes = np.sqrt(np.maximum(np.linalg.eigvalsh(predicted_shapes[rotor_id]), 1e-9))*2
                        axis_ratios.append(np.array([observed['minor'], observed['major']])/predicted_axes)
                    if np.min(axis_ratios) < .65 or np.max(axis_ratios) > 1.5:
                        good = False; out['previous_model_shape_rejected'] = True
                if good:
                    completed = np.c_[expected, np.ones(4)] @ A.T
                    missing = list(set(range(4)) - set(ids))
                    out.update(observed_ids=list(ids), inferred_ids=missing,
                               completion='recent_model_affine_transport', confidence=.6)
                    shapes = predicted_shapes if self.shapes is not None else np.repeat(np.mean([ellipse_tensor(r) for r in rotors], axis=0)[None], 4, axis=0)
                    for observed, rotor_id in zip(rotors, ids): shapes[rotor_id] = ellipse_tensor(observed)
            if completed is None:
                inferred = three_affine_hypotheses(rotors); out['three_hypotheses'] = inferred
                if not inferred['valid']:
                    out.update(state='three_unresolved', reason=inferred['reason']); return out
                completed = np.vstack([p, inferred['inferred_center']])
                out.update(observed_ids=[0, 1, 2], inferred_ids=[3],
                           completion='three_ellipse_metric_weak_perspective', confidence=.45)
                shapes = np.array([ellipse_tensor(r) for r in rotors]+[np.mean([ellipse_tensor(r) for r in rotors], axis=0)])
            out.update(state='three_completed', reason='position_completed_frequency_not_observed',
                       model_centers=completed.tolist(), axis_xy=normal.tolist(),
                       sign_prior='image_up_assumption', inferred_center_is_measurement=False)
            hubs = completed
        self.hubs = hubs.copy(); self.pose_t = t
        self.shapes = shapes.copy()
        self.body = np.array(body['center']) if body.get('detected') else None
        return out


def rotor_rates(q, rotors, t, velocity, w, unique_pixels=False):
    """Unique current pixel votes in independently measured rotor regions."""
    if len(q):
        if not unique_pixels:
            pixel = q['y'].astype(np.int64) * w + q['x']
            _, index = np.unique(pixel[::-1], return_index=True)
            q = q[len(q) - 1 - index]
        q = q[q['t'] >= round(t * 1e6) - 20000]
    xy = np.c_[q['x'], q['y']].astype(float)
    if len(q): xy += (t - q['t'] / 1e6)[:, None] * np.asarray(velocity)
    membership = []
    for r in rotors:
        d = xy - [r['cx'], r['cy']]; a = np.deg2rad(r['angle'])
        u = d[:, 0]*np.cos(a) + d[:, 1]*np.sin(a)
        v = -d[:, 0]*np.sin(a) + d[:, 1]*np.cos(a)
        membership.append((u / max(r['major']*.55, 1))**2 +
                          (v / max(r['minor']*.65, 2))**2 <= 1)
    ambiguous = np.sum(membership, axis=0) > 1 if membership else np.zeros(len(q), bool)
    result = []
    for inside in membership:
        f = q['frequency'][inside & ~ambiguous].astype(float)
        summary = dict(valid=False, pixels=len(f), frequency_hz=None,
                       mechanical_rpm_measured=False, overlapping_pixel_votes_excluded=int(np.sum(inside & ambiguous)))
        if len(f) >= PARAMS['frequency_min_pixels']:
            bins = np.floor(np.log(f / 60.) / np.log(1.04)).astype(int)
            hist = np.bincount(np.maximum(bins, 0)); k = int(np.argmax(hist))
            chosen = abs(bins - k) <= 1; fraction = float(chosen.mean())
            hz = float(np.median(f[chosen]))
            summary.update(valid=fraction >= PARAMS['frequency_mode_fraction'],
                           frequency_hz=hz, mode_fraction=fraction,
                           conditional_rpm_if_two_blades=hz*30.,
                           blade_count_assumed=2, harmonic_identity_unverified=True)
        result.append(summary)
    return result


class ThrustProxy:
    def __init__(self):
        self.reference = None; self.last = {}; self.last_full_t = None

    def step(self, t, geometry, rates):
        n = geometry['observed_count']; state = geometry['state']
        out = dict(valid=False, reason='no_qualified_geometry', actual_force_N=None,
                   thrust_direction_candidates_xy=None, physical_parameters_identified=False)
        if state not in ('four', 'three_completed', 'two_side'): return out
        axis = geometry.get('axis_xy')
        if axis is not None:
            out['thrust_direction_candidates_xy'] = [axis, (-np.asarray(axis)).tolist()]
        observed = [q['frequency_hz'] for q in rates if q.get('valid')]
        # Two side components need not correspond one-to-one to physical motors.
        # Never let their temporary indices overwrite the four-hub rate memory.
        if n >= 3:
            for rotor_id, rate in zip(geometry['observed_ids'], rates):
                if rate.get('valid'): self.last[int(rotor_id)] = (t, rate['frequency_hz'])
        if len(observed) != n:
            out['reason'] = 'not_every_visible_component_has_a_period'; return out
        f = list(observed); provenance = ['current_periodic_measurement'] * n
        if n == 3:
            rotor_id = geometry['inferred_ids'][0]; old = self.last.get(rotor_id)
            if old and t - old[0] <= PARAMS['missing_rate_max_age_s']:
                f.append(old[1]); provenance.append('past_rate_held_not_measured')
            else:
                f.append(float(np.median(observed))); provenance.append('hidden_equal_speed_assumption')
        elif n == 2:
            f += [float(np.median(observed))] * 2
            provenance += ['hidden_equal_speed_assumption'] * 2
        total = float(np.sum(np.square(f)))
        if self.reference is None: self.reference = total
        ratio = total / max(self.reference, 1.)
        out.update(valid=True, reason='conditional_relative_thrust_proxy',
                   observed_frequency_count=n, frequency_hypotheses_hz=f,
                   frequency_provenance=provenance, total_f2_proxy=total,
                   relative_f2_proxy=float(ratio), reference_f2=self.reference,
                   all_four_rates_observed=n == 4,
                   hidden_speed_assumption_used=any('assumption' in v for v in provenance),
                   merged_side_components_may_contain_multiple_motors=n == 2,
                   direction_sign_assumption='image_up',
                   differential_speed_changes_torque_not_parallel_thrust_direction=True)
        return out


class CausalPhysicsExpert:
    """Effective pixel thrust gain + constant disturbance, delayed position updates.

    Baseline and physical predictions are scored only when their target time has
    arrived. The experimental fused forecast takes at most 25% physical input.
    This estimates pixel-space combinations, never separately mass, cf or wind.
    """
    def __init__(self):
        self.theta = np.array([200., 0., 200.]); self.P = np.diag([1e5, 1e5, 1e5])
        self.queue = deque(); self.forecast_queue = deque()
        self.losses = {h: deque() for h in (.1, .2, .4)}; self.updates = 0

    def step(self, t, body, geometry, proxy, coefficient, baseline_predictions):
        out = dict(active=False, updates=self.updates, physical=None, fused=baseline_predictions,
                   weight=0., weights_by_horizon={str(h): 0. for h in (.1, .2, .4)},
                   parameters_effective_pixel_units=self.theta.tolist())
        if coefficient is None: return out
        p = np.asarray(coefficient['position'], float); v = np.asarray(coefficient['velocity'], float)
        while self.queue and self.queue[0]['target_t'] <= t + 1e-8:
            old = self.queue.popleft()
            if abs(t - old['target_t']) > .011 or not body.get('detected'): continue
            measured = np.asarray(body['center'], float)
            target = np.clip(2*(measured - old['p'] - old['v']*.2)/.2**2, -1500, 1500)
            X = old['X']; residual = np.clip(target - X @ self.theta, -600, 600)
            S = np.eye(2)*40000 + X @ self.P @ X.T
            K = np.linalg.solve(S, X @ self.P).T
            self.theta += K @ residual
            self.theta[0] = np.clip(self.theta[0], 0., 5000.)
            self.theta[1:] = np.clip(self.theta[1:], -2500., 2500.)
            self.P = (self.P - K @ X @ self.P) / .998
            self.P = (self.P + self.P.T)*.5
            self.updates += 1
        # Each forecast horizon must earn its own weight. A good .2s model is
        # not evidence that its .4s extrapolation is good.
        due = [q for q in self.forecast_queue if q['target_t'] <= t + 1e-8]
        self.forecast_queue = deque(q for q in self.forecast_queue if q['target_t'] > t + 1e-8)
        for old in due:
            if abs(t-old['target_t']) > .011 or not body.get('detected'): continue
            measured = np.asarray(body['center'], float)
            self.losses[old['h']].append((t, float(np.linalg.norm(old['baseline']-measured)),
                                        float(np.linalg.norm(old['physical']-measured))))
        for losses in self.losses.values():
            while losses and t-losses[0][0] > 1.2: losses.popleft()
        out.update(updates=self.updates, parameters_effective_pixel_units=self.theta.tolist())
        if not proxy.get('valid') or geometry.get('axis_xy') is None: return out
        feature = np.asarray(geometry['axis_xy']) * np.clip(proxy['relative_f2_proxy'], .2, 5.)
        X = np.array([[feature[0], 1., 0.], [feature[1], 0., 1.]])
        acceleration = np.clip(X @ self.theta, -1500., 1500.)
        physical = {}
        for h in (.1, .2, .4):
            displacement = v*h + .5*acceleration*h*h
            physical[str(h)] = (p + max(.5, 1-.9*h*h)*displacement).tolist()
        self.queue.append(dict(target_t=t+.2, X=X, p=p.copy(), v=v.copy(),
                               baseline=np.asarray(baseline_predictions['0.2']),
                               physical=np.asarray(physical['0.2'])))
        weights = {}
        for h, losses in self.losses.items():
            weight = 0.
            if self.updates >= 10 and len(losses) >= 10:
                b = np.mean([q[1] for q in losses]); f = np.mean([q[2] for q in losses])
                weight = min(.25, max(0., (b-f)/max(b, 1.))) * geometry['confidence']
                if proxy['hidden_speed_assumption_used']: weight *= .5
            weights[str(h)] = float(weight)
            self.forecast_queue.append(dict(target_t=t+h, h=h,
                baseline=np.asarray(baseline_predictions[str(h)]), physical=np.asarray(physical[str(h)])))
        fused = {h: ((1-weights[h])*np.asarray(point) + weights[h]*np.asarray(physical[h])).tolist()
                 for h, point in baseline_predictions.items()}
        out.update(active=max(weights.values()) > 0, updates=self.updates, physical=physical, fused=fused,
                   weight=max(weights.values()), weights_by_horizon=weights, acceleration_xy=acceleration.tolist(),
                   parameters_effective_pixel_units=self.theta.tolist(),
                   adaptation_uses='only body observations at already matured .2s targets',
                   force_calibrated=False)
        return out


class BranchObserver:
    def __init__(self, w, h, threshold=4., adaptive=False):
        self.w, self.h = w, h
        self.front = ThresholdFrontend(w, h, threshold, adaptive)
        self.line_front = ThresholdFrontend(w, h, min(threshold, 2.), False)
        self.split_front = ThresholdFrontend(w, h, threshold*PARAMS['split_threshold_factor'], False)
        self.geometry = BranchGeometry(); self.proxy = ThrustProxy(); self.expert = CausalPhysicsExpert()
        self.engine = native79.engine(w, h, 'strict7')
        self.output = self.engine.get_empty_output_buffer()
        self.mask = np.zeros((h, w), np.uint8)
        self.last_body = None

    def warmup(self, events):
        native79.checked_process(self.engine, events, self.output, self.w, self.h)

    def step(self, events, t, prepared, body, coefficient, baseline_predictions):
        if body.get('detected'):
            if self.last_body is not None and t - self.last_body > .5:
                self.geometry = BranchGeometry(); self.proxy = ThrustProxy(); self.expert = CausalPhysicsExpert()
            self.last_body = t
        q = native79.checked_process(self.engine, events, self.output, self.w, self.h)
        # Existing runner normalized its ON/OFF pair to 10 ms. Restore actual
        # counts here, so the user threshold always means a 20 ms batch.
        actual = dict(total=prepared['total'], on=prepared['on']*2, off=prepared['off']*2)
        self.front.set_hint(events, body, t)
        observation, _ = self.front.detect_prepared(events, t, actual)
        rotors = list(observation.get('rotors', [])); self.mask = self.front.mask146
        from boundary129 import bounded
        rotors, boundary = bounded(rotors, self.w, self.h)
        selected_front = self.front
        split = dict(valid=False, reason='not_three_components')
        if len(rotors) == 3:
            self.split_front.set_hint(events, body, t)
            higher, _ = self.split_front.detect_prepared(events, t, actual)
            higher_rotors, higher_boundary = bounded(higher.get('rotors', []), self.w, self.h)
            split = split_provenance(rotors, higher_rotors)
            if split['valid']:
                rotors, boundary, selected_front = higher_rotors, higher_boundary, self.split_front
        geometry = self.geometry.step(t, body, rotors)
        if geometry['state'] not in ('four', 'three_completed'):
            # A lower-density pre-ellipse path protects thin side signatures.
            # It is admitted only when *both* recovered components pass the
            # geometric side-view test. We do not choose whichever count is 2.
            self.line_front.set_hint(events, body, t)
            thin_observation, _ = self.line_front.detect_prepared(events, t, actual)
            thin_rotors, thin_boundary = bounded(thin_observation.get('rotors', []), self.w, self.h)
            line = two_axis(thin_rotors)
            if line['valid']:
                rotors, boundary = thin_rotors, thin_boundary
                geometry = self.geometry.step(t, body, rotors)
                selected_front = self.line_front
        self.mask = selected_front.mask146
        velocity = (coefficient or {}).get('velocity', [0., 0.])
        rates = rotor_rates(q, rotors, t, velocity, self.w)
        proxy = self.proxy.step(t, geometry, rates)
        expert = self.expert.step(t, body, geometry, proxy, coefficient, baseline_predictions)
        raw_active = int(np.sum((actual['on'] + actual['off']) > 0))
        kept = int(np.count_nonzero(self.mask))
        return dict(geometry=geometry, rates=rates, thrust_proxy=proxy, expert=expert,
                    threshold=selected_front.support_threshold, threshold_units=PARAMS['threshold_units'],
                    active_pixels=raw_active, retained_pixels=kept, boundary=boundary,
                    input_events=len(events), retained_display_events=int(np.sum(self.mask[events['y'], events['x']] > 0)),
                    selected_mask='thin_fallback' if selected_front is self.line_front else ('split_recovery' if selected_front is self.split_front else 'main'),
                    split_recovery=split,
                    roi=selected_front.current_roi, adaptive=selected_front.adaptive146,
                    feature_suppression_is_display_and_geometry_only=True,
                    temporal_period_estimator_sees_original_events=True)


def render_panel(image, observer, measurement, t):
    """Native ON/OFF and suppressed support, with inferred hubs visibly distinct."""
    h, w = image.shape[:2]
    right = np.full_like(image, 255)
    right[observer.mask > 0] = (0, 0, 0)
    geometry = measurement['geometry']
    for rotor in geometry['observed_rotors']:
        center = tuple(np.rint([rotor['cx'], rotor['cy']]).astype(int))
        axes = tuple(np.maximum(1, np.rint([rotor['major']/2, rotor['minor']/2]).astype(int)))
        cv2.ellipse(right, center, axes, rotor['angle'], 0, 360, (20, 160, 20), 2)
        cv2.circle(right, center, 3, (20, 160, 20), -1)
    for index in geometry['inferred_ids']:
        center = tuple(np.rint(geometry['model_centers'][index]).astype(int))
        cv2.circle(right, center, 10, (180, 0, 180), 2)
        cv2.putText(right, 'inferred', (center[0]+12, center[1]), 0, .5, (180, 0, 180), 1)
    if geometry.get('axis_xy') is not None:
        origin = np.mean(geometry['model_centers'], axis=0)
        normal = unit(geometry['axis_xy'])
        for sign in (-1, 1):
            cv2.arrowedLine(right, tuple(np.rint(origin).astype(int)),
                           tuple(np.rint(origin+sign*40*normal).astype(int)), (210, 100, 0), 2)
    # Place labels outside the data: targets at the sensor boundary must not be
    # covered by a title bar. The crop is a display-only current-body ROI.
    roi = measurement.get('roi') or [0, 0, w, h]
    x1, y1, x2, y2 = np.asarray(roi, int)
    x1, y1 = max(0, x1-8), max(0, y1-8)
    x2, y2 = min(w, x2+8), min(h, y2+8)
    panel = np.full((430, 1280, 3), 255, np.uint8)
    for offset, source in ((0, image), (640, right)):
        crop = source[y1:y2, x1:x2]
        scale = min(620/max(1, crop.shape[1]), 350/max(1, crop.shape[0]))
        resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        panel[70:70+resized.shape[0], offset+10:offset+10+resized.shape[1]] = resized
    cv2.putText(panel, f'Native ON/OFF ROI t={t:.2f}s', (15, 25), 0, .65, (0, 0, 0), 2)
    cv2.putText(panel, f'Threshold {measurement["threshold"]:.2f} pairs/20ms; {geometry["state"]}',
                (655, 25), 0, .6, (0, 0, 0), 2)
    cv2.putText(panel, 'Green: measured; purple: inferred; arrows: unsigned normal',
                (655, 50), 0, .45, (0, 0, 0), 1)
    return panel
