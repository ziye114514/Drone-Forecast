"""Attribute the dominant spatial-variance component to one current rotor slot.

Handoff question (2026-10-29): at 26.499735 s the axis spatial variance has one
dominant component (231.875 Hz) while four rotor slots report
303.25/230.21/256.48/284.43 Hz. One dominant component is not the whole body:
it must not be squared and called the four-rotor thrust. A moving dominant peak
also has two different causes - the same rotor changing speed, or a different
rotor dominating the variance amplitude.

This module only audits that attribution on the current observation. It builds
no motion predictor, converts no event frequency into newton thrust, and never
reports a hidden rotor frequency as measured. Slot identity always comes from
the existing VisibilityPeriodProxy mapper; the variance spectrum is used only
as an independent observation of whichever slot it actually matches.
"""
import collections
import numpy as np
from scipy.signal import lombscargle

GRID = np.arange(60., 1200.0001, 3.125)


def power_spectrum(times, values, valid):
    """Detrended Lomb-Scargle power on the grid used by the earlier probe."""
    valid = np.asarray(valid, bool)
    info = dict(bins=int(valid.sum()), coverage=float(np.mean(valid)) if len(valid) else 0.)
    if len(valid) < 64 or info['coverage'] < .6:
        return None, dict(info, reason='sparse_moment_bins')
    t = np.asarray(times, float)[valid]
    y = np.asarray(values, float)[valid]
    scale = max(1., float(np.mean(abs(y))))
    if np.std(y) <= 1e-7 * scale:
        return None, dict(info, reason='constant_spatial_moment')
    t = t - t[0]
    design = np.column_stack((np.ones(len(t)), t))
    y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if not np.isfinite(y).all() or np.std(y) <= 0:
        return None, dict(info, reason='degenerate_residual')
    power = lombscargle(t, y, 2 * np.pi * GRID, normalize='normalize', floating_mean=True)
    return np.asarray(power, float), info


def peaks(power, top_k=4, min_ratio=8.):
    """Local maxima sorted by power; no peak is promoted without margin."""
    if power is None or not len(power):
        return []
    median = max(float(np.median(power)), 1e-12)
    inner = power[1:-1]
    found = np.flatnonzero((inner > power[:-2]) & (inner >= power[2:])) + 1
    if not len(found):
        return []
    step = float(GRID[1] - GRID[0])
    result = []
    for k in found[np.argsort(power[found])[::-1][:top_k]]:
        low, mid, high = (np.log(max(power[max(k - 1, 0)], 1e-12)),
                          np.log(max(power[k], 1e-12)),
                          np.log(max(power[min(k + 1, len(power) - 1)], 1e-12)))
        den = low - 2 * mid + high
        shift = 0. if den >= -1e-12 else float(np.clip(.5 * (low - high) / den, -.5, .5))
        ratio = float(power[k] / median)
        if ratio < min_ratio:
            continue
        result.append(dict(frequency_hz=float(GRID[k] + shift * step), grid_hz=float(GRID[k]),
            normalized_power=float(power[k]), peak_to_median=ratio, sub_bin_shift=shift))
    return result


def _log_ratio(a, b):
    return float(abs(np.log(a / b)))


def attribute(peak, slots, rel_tol=.03):
    """Match one peak to the CURRENT slot set, keeping harmonics non-identity.

    modes: owned / ambiguous / harmonic_of_slot / half_of_slot / unowned.
    """
    slots = {int(k): float(v) for k, v in (slots or {}).items() if np.isfinite(v) and v > 0}
    f = float(peak['frequency_hz'])
    if not slots:
        return dict(mode='unowned', owner_model_id=None, reason='no_current_slot_claims',
            harmonic_of=None, half_of=None)
    direct = sorted(((i, _log_ratio(f, v)) for i, v in slots.items()), key=lambda x: x[1])
    gate = float(np.log1p(rel_tol))
    inside = [(i, e) for i, e in direct if e <= gate]
    if len(inside) >= 2:
        return dict(mode='ambiguous', owner_model_id=None, reason='two_current_slots_inside_tolerance',
            candidates=[dict(model_id=i, log_error=e) for i, e in inside],
            margin_log=float(inside[1][1] - inside[0][1]))
    if len(inside) == 1:
        owner = inside[0][0]
        others = [e for i, e in direct if i != owner]
        return dict(mode='owned', owner_model_id=owner, reason='peak_matches_one_current_slot',
            log_error=float(inside[0][1]), relative_error=float(np.expm1(inside[0][1])),
            margin_log=float(min(others) - inside[0][1]) if others else None)
    # A harmonic of an observed rotor is not that rotor's current frequency.
    for multiple, mode in ((2., 'harmonic_of_slot'), (3., 'harmonic_of_slot')):
        hits = [(i, _log_ratio(f, multiple * v)) for i, v in slots.items()
                if _log_ratio(f, multiple * v) <= float(np.log1p(rel_tol))]
        if hits:
            i, e = min(hits, key=lambda x: x[1])
            return dict(mode=mode, owner_model_id=None, reason='peak_is_multiple_of_slot_rotation',
                harmonic_of=i, multiple=multiple, log_error=float(e))
    hits = [(i, _log_ratio(2 * f, v)) for i, v in slots.items()
            if _log_ratio(2 * f, v) <= float(np.log1p(rel_tol))]
    if hits:
        i, e = min(hits, key=lambda x: x[1])
        return dict(mode='half_of_slot', owner_model_id=None,
            reason='peak_is_half_of_slot_rotation_blade_count_unverified', half_of=i, log_error=float(e))
    return dict(mode='unowned', owner_model_id=None, reason='peak_matches_no_current_slot',
        nearest_slot=direct[0][0], nearest_log_error=float(direct[0][1]))


def assign(found, slots, rel_tol=.03):
    """One peak per slot: a weaker duplicate cannot corroborate the same rotor."""
    free = set(int(k) for k in (slots or {}))
    out = []
    for peak in found:
        info = attribute(peak, slots, rel_tol)
        independent = False
        if info['mode'] == 'owned' and info['owner_model_id'] in free:
            free.discard(info['owner_model_id'])
            independent = True
        out.append(dict(info, peak=peak, independent=independent))
    return out


class PeakOwnershipAudit:
    """Causal state machine over consecutive current observations.

    The decision rule compares the peak with BOTH the previous peak and the
    previous slot set, so a stable slot set plus a moving dominant peak is
    labelled a dominance change, not a rotation change.
    """
    max_gap_s = .5
    link_max_s = .35

    def __init__(self, rel_tol=.03, top_k=4, min_ratio=8., switch_tol=.03):
        self.rel_tol = rel_tol
        self.top_k = top_k
        self.min_ratio = min_ratio
        self.switch_tol = switch_tol
        self.reset()

    def reset(self):
        self.last_t = None
        self.previous = None
        self.counts = collections.Counter()

    def _classify(self, t, dominant, slots):
        prev = self.previous
        if dominant is None:
            return 'no_periodic_component', 'moment_unqualified_or_no_peak'
        mode = dominant['mode']
        if mode == 'ambiguous':
            return 'ambiguous_ownership', dominant['reason']
        if mode in ('harmonic_of_slot', 'half_of_slot'):
            return 'harmonic_not_identity', dominant['reason']
        if mode == 'unowned':
            if not slots:
                return 'no_identified_slot_claims', 'variance_peak_without_current_slot_set'
            return 'unowned_peak', dominant['reason']
        if prev is None:
            return 'owner_initialized', 'first_current_attribution'
        if t - prev['t'] > self.link_max_s:
            return 'owner_reinitialized', 'previous_attribution_too_old_to_link'
        common = set(prev['slots']) & set(slots)
        set_stable = bool(common) and all(
            _log_ratio(slots[i], prev['slots'][i]) <= self.rel_tol for i in common)
        same = prev['owner'] == dominant['owner_model_id']
        moved = _log_ratio(dominant['peak']['frequency_hz'], prev['frequency']) > self.switch_tol
        if same and not moved:
            return 'stable_owner', 'same_slot_and_same_frequency'
        if same and moved:
            if set_stable:
                return 'peak_moved_without_slot_change', 'claimed_slot_unchanged_so_shift_is_not_rotation'
            return 'rotation_change_of_same_owner', 'same_slot_frequency_changed'
        if set_stable and moved:
            return 'dominance_switch', 'dominant_component_jumped_to_another_stable_slot'
        if set_stable:
            return 'relabeled_same_frequency', 'owner_label_changed_at_unchanged_frequency'
        return 'owner_change_with_unstable_slots', 'slot_set_changed_with_owner_change'

    def step(self, t, times, values, valid, slots):
        if self.last_t is not None and t <= self.last_t:
            raise ValueError('Increasing causal time required')
        if self.last_t is not None and t - self.last_t > self.max_gap_s:
            self.reset()
        self.last_t = t
        slots = {int(k): float(v) for k, v in (slots or {}).items()}
        power, info = power_spectrum(times, values, valid)
        found = peaks(power, self.top_k, self.min_ratio) if power is not None else []
        assignments = assign(found, slots, self.rel_tol)
        corroborated = sorted({a['owner_model_id'] for a in assignments if a['independent']})
        dominant = assignments[0] if assignments else None
        state, reason = self._classify(t, dominant, slots)
        self.counts[state] += 1
        result = dict(t=float(t), state=state, reason=reason, claimed_slots=sorted(slots),
            slot_frequencies={str(i): float(v) for i, v in sorted(slots.items())},
            peaks=found, attribution=assignments, corroborated_slots=corroborated,
            resolved_slot_fraction=len(corroborated) / max(len(slots), 1),
            dominant_model_id=dominant['owner_model_id'] if dominant else None,
            dominant_frequency_hz=dominant['peak']['frequency_hz'] if dominant else None,
            moment_bins=info.get('bins'), moment_coverage=info.get('coverage'),
            spectrum_qualified=power is not None,
            mechanical_RPM_measured=False, thrust_measured=False,
            harmonic_identity_unverified=True, hidden_frequencies_measured=0)
        if dominant is not None and dominant['mode'] == 'owned':
            self.previous = dict(owner=dominant['owner_model_id'],
                frequency=dominant['peak']['frequency_hz'], slots=dict(slots), t=float(t))
        return result
