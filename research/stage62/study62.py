"""Frozen same-input causal replay: independent axis prediction and learning."""
from pathlib import Path
import argparse, copy, datetime, hashlib, html, importlib.util, inspect, json, os, sys
import numpy as np
from independent62 import *

BASE = ROOT.parent / 'output'
OUT = BASE / 'stage62_independent_axis_20261021'
REFERENCE = BASE / 'stage61_preellipse_lines_20261021/forecast'
CASES = ['clear6', 'side6', 'train3', 'negative5']
PREFIXES = dict(side6=225, train3=60)
HISTORY = ['hold', 'cv', 'cv_hist_acc', 'bounded_acc', 'huber_acc', 'trusted_acc']
FREQUENCY = ['nominal_axis_freq', 'adaptive_axis_freq', 'adaptive_fixed_axis_freq']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s]
def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                          default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()), encoding='utf-8')


def sources():
    old = load(REFERENCE / 'protocol.json')['source_hashes']
    result = {n: sha(ROOT/n) for n in old}
    assert result == old, 'Frozen baseline sources changed'
    for name in ['independent62.py', 'study62.py']:
        result['research/stage62/'+name] = sha(Path(__file__).with_name(name))
    return result


def inputs():
    return {c: sha(REFERENCE/f'off_{c}/observations_forecasts.jsonl') for c in CASES}


def checked_step(managed, selector, row):
    q, lifecycle = managed.step(row['t'], row['observation'], row['sensing'], row['slot_audit']['calibration_epoch'])
    q = selector.step(q, row['observation'])
    q['association_diagnostic'] = row['frontends']['delayed_feedback']['association_diagnostic']
    return q, lifecycle


def controls():
    # One geometry observation, no frequency at any time: independent admission
    # must work, while missing position must never launch/consume feedback.
    from event_drone_forecast.paper_physics import unit
    predictors = {m: Predictor62(m) for m in METHODS}
    no_position = Predictor62('axis_learn')
    active = {m: 0 for m in METHODS}
    for j in range(100):
        t = j*.02
        rotor = dict(cx=100., cy=80., major=40., minor=12., ratio=.3,
                     angle=float(10*np.sin(t*2)), geometry_quality=.8,
                     activity=100., observed_now=True, axis_only=False)
        sensing = dict(rotors=[rotor], four_visible=False, four_frequencies=False,
                       thrust_ratio=None, frequency_quality=0.)
        obs = dict(detected=True, center=[10*t*t, -8*t*t], confidence=.9, geometry_quality=0.)
        row = dict(t=t, sensing=sensing, observation=obs)
        for m, predictor in predictors.items():
            q, _ = predictor.step(row)
            active[m] += q['feature_diagnostic']['models']['adaptive_axis_only']['active']
        no_position.step(dict(row, observation=dict(obs, detected=False, center=None)))
    assert active['shared'] == 0 and active['axis_gate'] > 0 and active['axis_learn'] > 0
    assert predictors['shared'].models['adaptive_axis_only'].updates == 0
    assert predictors['axis_gate'].models['adaptive_axis_only'].updates == 0
    assert predictors['axis_learn'].models['adaptive_axis_only'].updates > 0
    assert no_position.models['adaptive_axis_only'].updates == 0
    before = predictors['axis_learn'].models['adaptive_axis_only'].updates
    # Expired axis: same CV fallback, no new learning from stale state.
    sensing = dict(sensing, rotors=[])
    for j in range(100, 125):
        t = j*.02
        q, _ = predictors['axis_learn'].step(dict(t=t, sensing=sensing, observation=obs))
    assert not q['feature_diagnostic']['models']['adaptive_axis_only']['active']
    assert q['coefficients']['adaptive_axis_only'] == q['coefficients']['cv']
    assert not predictors['axis_learn'].models['adaptive_axis_only'].queue
    return dict(single_geometry_no_frequency_active_ticks=active,
                independent_axis_updates_before_expiry=before,
                no_position_updates=0, expired_axis_returns_same_CV=True,
                missing_rotors_do_not_launch_new_regressors=True,
                thresholds_unchanged=True)


def freeze():
    assert not OUT.exists()
    checks = controls()
    protocol = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    source_hashes=sources(), input_hashes=inputs(), parameters=PARAMS,
                    cases=CASES, methods=METHODS, models=MODELS, controls=checks,
                    derived_step_sha256=hashlib.sha256(step_source().encode()).hexdigest(),
                    label_hashes=load(REFERENCE/'protocol.json')['label_hashes'],
                    design='700 saved current inputs from actual61 RAW runs, replayed causally. '
                           'Baseline full15/lifecycle exact61; all methods same six histories, '
                           'three frequency experts and CV fallback. Predictions before sparse labels. '
                           'No new RAW detector execution or runtime claim from this replay.',
                    prior_art=['https://arxiv.org/abs/2604.18289',
                               'https://www.mathworks.com/help/ident/ug/online-recursive-least-squares-estimation.html'],
                    actual_reuse='Original27 predictor source,28 canonical axis,51 frequency '
                                 'reference/lifecycle,52 delayed RLS,29 selector,22 scoring. '
                                 'Only axis-only branch admission changes; standard missing-modality '
                                 'handling, not a new algorithm claim.')
    save(OUT/'protocol.json', protocol)
    (OUT/'derived_step.py').write_text(step_source(), encoding='utf-8')
    print('62 protocol frozen; controlled missing-frequency/position/expiry passed', flush=True)


def verify():
    p = load(OUT/'protocol.json')
    assert p['source_hashes'] == sources() and p['input_hashes'] == inputs()
    assert p['derived_step_sha256'] == hashlib.sha256(step_source().encode()).hexdigest()
    return p


def guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name == 'targets.json' or path.is_relative_to((BASE/'raw_only_project/evaluation_data').resolve()):
            raise RuntimeError('Truth forbidden during inference: '+str(path))


def infer():
    verify(); sys.addaudithook(guard)
    summary = []
    for case in CASES:
        rr = rows(REFERENCE/f'off_{case}/observations_forecasts.jsonl')
        models = {m: Managed62(m) for m in METHODS}; selectors = {m: Selector() for m in METHODS}
        folder = OUT/case; folder.mkdir(exist_ok=False)
        stats = {m: dict(active=0, no_four_active=0, no_frequency_active=0, updates=0) for m in METHODS}
        with (folder/'forecasts.jsonl').open('x', encoding='utf-8') as f:
            for row in rr:
                forecasts = {}; lifecycle = {}; audits = {}
                for method in METHODS:
                    q, d = checked_step(models[method], selectors[method], row)
                    old = row['frontends']['delayed_feedback']
                    assert q['valid'] == old['valid']
                    if method == 'shared':
                        assert q == old, (case, row['t'], 'full15 baseline regression')
                        assert d == row['lifecycle']['delayed_feedback'], (case, row['t'], 'lifecycle regression')
                    for name in HISTORY+FREQUENCY+['fixed_history_mix', 'adaptive_history_mix']:
                        assert q['coefficients'].get(name) == old['coefficients'].get(name), (case, row['t'], method, name)
                    if method == 'axis_gate':
                        a=models[method].predictor.models['adaptive_axis_only']; b=models['shared'].predictor.models['adaptive_axis_only']
                        assert a.updates == b.updates
                        np.testing.assert_array_equal(a.parameters, b.parameters)
                        np.testing.assert_array_equal(a.cov, b.cov)
                        np.testing.assert_array_equal(a.disturbance, b.disturbance)
                    md=q['feature_diagnostic']['models']['adaptive_axis_only']; s=stats[method]
                    s['active'] += md['active']; s['no_four_active'] += md['active'] and not row['sensing']['four_visible']
                    s['no_frequency_active'] += md['active'] and not q['feature_diagnostic']['active']; s['updates']=md['updates']
                    forecasts[method]=q; lifecycle[method]=d; audits[method]=diagnostics(models[method], row['t'], row['sensing'])
                f.write(json.dumps(dict(t=row['t'], frontends=forecasts, lifecycle=lifecycle, axis_audit=audits), allow_nan=False)+'\n')
        save(folder/'completion.json', dict(frames=len(rr), labels_read=False, shared_complete15_exact61=True,
                                          histories_frequency_experts_and_history_mixes_exact61=True,
                                          same_CV_fallback=True, methods=stats))
        summary.append(dict(case=case, frames=len(rr), methods=stats)); print(summary[-1], flush=True)
    save(OUT/'replay_checks.json', dict(table=summary, parent_input_ticks=700,
                                      method_steps=2100, new_RAW_reads=False))
    verify()


def prefix():
    verify(); sys.addaudithook(guard); summary=[]
    for case, count in PREFIXES.items():
        rr=rows(REFERENCE/f'off_{case}/observations_forecasts.jsonl')[:count]
        expected=rows(OUT/case/'forecasts.jsonl')[:count]
        for method in METHODS:
            model=Managed62(method); selector=Selector()
            for row, old in zip(rr, expected):
                q,d=checked_step(model, selector, row)
                assert q==old['frontends'][method] and d==old['lifecycle'][method]
                assert diagnostics(model,row['t'],row['sensing'])==old['axis_audit'][method]
        summary.append(dict(case=case, prefix_parent_ticks=count, all_three_complete15_and_state_exact=True))
    save(OUT/'prefix_checks.json', dict(table=summary, independent_reinitialization=True,
                                      actual_RAW_prefix=False, scope='Saved causal current-input prefixes only.'))
    print('62 causal saved-input prefixes passed', flush=True)


def score():
    verify()
    assert len(load(OUT/'prefix_checks.json')['table']) == 2
    for case in CASES: assert load(OUT/case/'completion.json')['shared_complete15_exact61']
    sys.path.insert(0,str(ROOT/'research/stage55')); import study55
    spec=importlib.util.spec_from_file_location('score62_original22',ROOT/'research/stage22/study22.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT=OUT;module.INPUT=OUT;module.CASES=CASES;module.FRONTENDS=METHODS;module.MODELS=MODELS;module.verify=verify;module.evaluate=prior.evaluate
    module.label_path=lambda c: BASE/'raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train5.json' if c=='negative5' else 'train6.json')
    module.score()
    new=load(OUT/'analysis.json');old=load(REFERENCE/'analysis.json')
    lookup={(r['case'].removeprefix('off_'),r['model'],r['nominal_h']):r for r in old['table'] if r['case'].startswith('off_')}
    baseline=0
    for r in new['table']:
        b=lookup[r['case'],r['model'],r['nominal_h']]
        expected=dict(b,case=r['case'],frontend=r['frontend'])
        if r['frontend']=='shared':assert r==expected,(r,b);baseline+=1
        for key in ['present_targets','anchors_available','valid_forecasts','absent_targets','known_negative_published_anchors']:
            assert r[key]==b[key]
    save(OUT/'score_checks.json',dict(rows=len(new['table']),baseline_rows_exact61=baseline,all_coverage_exact61=True))
    print('62 sparse future score saved; baseline scores exact61',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','infer','prefix','score'])
    globals()[p.parse_args().action]()
