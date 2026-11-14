"""Geometric identifiability, suppression, provenance and causal-update controls."""
from pathlib import Path
import json
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from branches146 import (BranchGeometry, ThrustProxy, CausalPhysicsExpert,
                         ThresholdFrontend, three_affine_hypotheses, two_axis, split_provenance)


def rotor(x, y, ratio=.5, angle=0.):
    return dict(cx=float(x), cy=float(y), major=32., minor=32.*ratio,
                angle=angle, ratio=ratio, mode='line' if ratio < .1 else 'ellipse',
                geometry_quality=.9, inferred_shape=False)


def main():
    controls = {}
    disks = [rotor(x, y) for x, y in [(80, 90), (200, 90), (200, 150), (80, 150)]]
    for missing in range(4):
        current = [r for i, r in enumerate(disks) if i != missing]
        completion = three_affine_hypotheses(current)
        assert completion['valid'], completion
        np.testing.assert_allclose(completion['inferred_center'],
                                   [disks[missing]['cx'], disks[missing]['cy']], atol=1e-8)
    controls['all_four_three_disk_completions_exact_under_affine_assumptions'] = True
    assert not three_affine_hypotheses([rotor(20, 50), rotor(50, 50), rotor(90, 50)])['valid']
    controls['collinear_three_not_completed'] = True
    assert not two_axis([rotor(20, 30), rotor(90, 30)])['valid']
    assert two_axis([rotor(20, 30, .05), rotor(90, 30, .05)])['valid']
    assert not two_axis([rotor(20, 30, .05, 90), rotor(90, 30, .05, 90)])['valid']
    controls['two_disks_require_flatness_and_axis_alignment'] = True
    model = BranchGeometry(); body = dict(detected=True, center=[140., 120.])
    full = model.step(.02, body, disks)
    assert full['state'] == 'four'
    trapezoid = [rotor(x, y) for x, y in [(70, 80), (210, 80), (180, 160), (100, 160)]]
    projective = BranchGeometry().step(.02, body, trapezoid)
    assert projective['state'] == 'four' and projective['diagonal_affine_error_px'] > 8.
    controls['perspective_trapezoid_not_rejected_as_non_parallelogram'] = True
    parents = [disks[0], disks[3], dict(rotor(200, 120), major=65., minor=40., ratio=40/65)]
    assert split_provenance(parents, disks)['valid']
    assert not split_provenance(parents, disks[:3])['valid']
    controls['split_requires_four_existing_children_with_one_local_parent_split'] = True
    for missing in range(4):
        current = [r for i, r in enumerate(disks) if i != missing]
        measured = model.step(.04 + missing*.02, body, current)
        assert measured['state'] == 'three_completed' and len(measured['inferred_ids']) == 1
        assert measured['observed_count'] == 3 and len(measured['observed_rotors']) == 3
    assert model.step(.14, body, disks[:1])['state'] == 'insufficient'
    controls['inferred_hub_never_counted_as_observation_and_one_is_allowed'] = True
    wrong = [dict(r) for r in disks[:3]]; wrong[2].update(minor=30., ratio=30/32)
    assert not three_affine_hypotheses(wrong)['valid']
    assert BranchGeometry().step(.02, body, wrong)['state'] == 'three_unresolved'
    controls['inconsistent_three_disk_shapes_not_blindly_completed'] = True
    front = ThresholdFrontend(80, 60, 2.)
    on = np.zeros((60, 80), np.float32); on[10:35, 20:50] = 12.
    assert not np.any(front.feature(on, np.zeros_like(on)))
    assert np.count_nonzero(front.support_mask(on, 123.)) > 0
    assert front.support_threshold == 2.
    assert not np.any(front.support_mask(on*.01, 0.))
    controls['explicit_threshold_overrides_old_quantile_monopolar_rejected'] = True
    proxy = ThrustProxy(); rates = [dict(valid=True, frequency_hz=f) for f in (100., 110., 120., 130.)]
    full['observed_ids'] = [0, 1, 2, 3]
    measured = proxy.step(.1, full, rates)
    assert measured['all_four_rates_observed']
    triple = dict(full, state='three_completed', observed_count=3, observed_ids=[0, 1, 2], inferred_ids=[3])
    held = proxy.step(.12, triple, rates[:3])
    assert held['frequency_provenance'][-1] == 'past_rate_held_not_measured'
    assert held['total_f2_proxy'] == measured['total_f2_proxy']
    expired = proxy.step(.4, triple, rates[:3])
    assert expired['hidden_speed_assumption_used'] and expired['actual_force_N'] is None
    before = dict(proxy.last)
    pair = dict(full, state='two_side', observed_count=2, observed_ids=[0, 1])
    proxy.step(.42, pair, rates[:2])
    assert proxy.last == before
    controls['missing_rate_held_then_assumed_never_fabricated_as_measured'] = True
    expert = CausalPhysicsExpert()
    coefficient = dict(position=[0., 0.], velocity=[10., 0.])
    baseline = {str(h): [10*h, 0.] for h in (.1, .2, .4)}
    feature = dict(valid=True, relative_f2_proxy=1., hidden_speed_assumption_used=False)
    geometry = dict(axis_xy=[0., -1.], confidence=.8)
    for t in (0., .02, .10, .18):
        result = expert.step(t, dict(detected=True, center=[10*t, 0.]), geometry, feature,
                             dict(coefficient, position=[10*t, 0.]), baseline)
        assert result['updates'] == 0 and result['weight'] == 0.
    expert.step(.2, dict(detected=True, center=[2., 0.]), geometry, feature,
                dict(coefficient, position=[2., 0.]), baseline)
    assert expert.updates == 1
    controls['no_adaptation_or_forecast_weight_before_matured_observation'] = True
    expert = CausalPhysicsExpert()
    for index in range(21):
        t = index*.02
        coefficient = dict(position=[10*t, 0.], velocity=[10., 0.])
        wrong_baseline = {str(h): [10*(t+h)+20, 0.] for h in (.1, .2, .4)}
        checked = expert.step(t, dict(detected=True, center=[10*t, 0.]), geometry, feature,
                              coefficient, wrong_baseline)
    assert checked['weights_by_horizon']['0.2'] > 0
    assert checked['weights_by_horizon']['0.4'] == 0
    controls['good_matured_point2_does_not_enable_unvalidated_point4_weight'] = True
    folder = Path(__file__).resolve().parents[3] / 'output/stage146_threshold_branches_20261024'
    folder.mkdir(exist_ok=True)
    (folder / 'controls.json').write_text(json.dumps(controls, indent=2), encoding='utf-8')
    print(json.dumps(controls), flush=True)


if __name__ == '__main__': main()
