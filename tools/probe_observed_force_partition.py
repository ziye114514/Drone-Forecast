"""Small causal mechanism and frozen-observation comparison; no RAW/GPU work.

Labels are opened only after every inference arm has been saved. The old
observer path and old bank must reproduce stage161 before comparing scores.
"""
from pathlib import Path
import argparse, copy, hashlib, json, sys, time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'core'),str(ROOT), str(ROOT/'tools')]
from rotor_thrust_bank import RotorThrustBankProxy, LegacyRotorThrustBankProxy
from rotor_force_observer import RotorForceObserver
from physical_publication import evaluate


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')


def synthetic(ids, frequency=300., axis=(0., -1.), epoch=0):
    hubs = [[-50., -40.], [50., -40.], [50., 40.], [-50., 40.]]
    structure = dict(mode={4:'four', 3:'three', 2:'partial_two'}[len(ids)],
        model_centers=hubs, observed_model_ids=list(ids), geometry_epoch=epoch,
        axis_xy=list(axis), geometric_weight=.7)
    rotors = [dict(cx=hubs[i][0], cy=hubs[i][1], major=30., ratio=.5,
        mode='ellipse', current_observation=True) for i in ids]
    rates = [dict(valid=True, frequency_hz=frequency) for i in ids]
    return structure, rotors, rates, dict(qualified=False)


def mechanism_checks():
    b = RotorThrustBankProxy(); m = RotorForceObserver(); log = []
    t = 0.
    # Starts from two observed motors, never from a four-motor initialization.
    for ids in [[0,1]]*7 + [[0,1,2,3]]*3 + [[0,1,2]]*3 + [[0,1]]*3 + [[0,1,2,3]]*3:
        t += 1/30
        u, q, meta = b.step(t, *synthetic(ids))
        c, valid, force, state = m.step(t, dict(detected=True, center=[0.,0.]), u, q,
            allow_gain_learning=meta['learning_input_current'], input_meta=meta)
        if valid:
            assert u is not None and len(meta['observed_components']) >= 2
            assert np.linalg.norm(c['acceleration']) < 1e-8
            assert np.linalg.norm(c['position']) < 1e-8
            assert np.linalg.eigvalsh(m.P).min() > -1e-6
        log.append(dict(t=t, ids=list(ids), active=force['active'],
            acceleration=c['acceleration'], transferred=force.get('transferred_input_u')))
    # Keep identities but change speed. No visibility bookkeeping may cancel it.
    t += 1/30; u, q, meta = b.step(t, *synthetic([0,1,2,3], frequency=315.))
    c, valid, force, _ = m.step(t, dict(detected=True, center=[0.,0.]), u, q,
        allow_gain_learning=False, input_meta=meta)
    speed_response = np.asarray(c['acceleration'])
    np.testing.assert_allclose(speed_response, [0., -20.5], atol=1e-7)
    assert not force['visibility_partition_changed']
    # Disappear and return without motion evidence: do not duplicate the force.
    t += 1/30
    c, _, force, _ = m.step(t, dict(detected=False), None, 0., input_meta={})
    np.testing.assert_allclose(c['acceleration'], speed_response, atol=1e-7)
    assert not force['active'] and np.linalg.norm(m.u) == 0
    t += 1/30
    c, _, force, _ = m.step(t, dict(detected=False), u, q, input_meta=meta)
    np.testing.assert_allclose(c['acceleration'], speed_response, atol=1e-7)
    # A normal rotation with unchanged visible motors must also remain physical.
    n = np.array([.1, -np.sqrt(.99)])
    u2 = np.linalg.norm(u)*n
    meta2 = copy.deepcopy(meta)
    meta2['observed_components'] = {k:(np.linalg.norm(v)*n).tolist()
        for k,v in meta['observed_components'].items()}
    t += 1/30
    c, _, _, _ = m.step(t, dict(detected=False), u2, q, allow_gain_learning=False, input_meta=meta2)
    np.testing.assert_allclose(c['acceleration'], speed_response+m.x[6]*(u2-u), atol=1e-7)
    # Unresolved side startup uses a zero-baseline change, never f^2 times four.
    side = RotorThrustBankProxy(); structure = dict(mode='edge_on', axis_xy=[0.,-1.])
    side.step(0., structure, [], [], dict(qualified=True, frequency_hz=300.))
    su, _, si = side.step(.033, structure, [], [], dict(qualified=True, frequency_hz=300.))
    np.testing.assert_allclose(su, [0.,0.])
    su, _, si = side.step(.066, structure, [], [], dict(qualified=True, frequency_hz=315.))
    np.testing.assert_allclose(su, [0.,-(315.**2-300.**2)/360000.])
    assert not si['learning_input_current'] and not si['mixed_proxy_is_total_thrust']
    # Same current prefix must be independent of an altered future measurement.
    def prefix(future):
        model = RotorThrustBankProxy(); result = []
        for j,f in enumerate([300.]*6+[future]):
            u, _, _ = model.step(j/30, *synthetic([0,1], frequency=f))
            result.append(None if u is None else u.tolist())
        return result[:-1]
    assert prefix(315.) == prefix(900.)
    # Moderate same-slot steps can remain current input; a harmonic-sized
    # jump is still unresolved. No future position is used for this response.
    abrupt_cases = {}
    for speed in [360.,600.]:
        abrupt_bank = RotorThrustBankProxy(); abrupt_model = RotorForceObserver(); abrupt = []
        for j, f in enumerate([300.]*8+[speed]*4):
            u, q, info = abrupt_bank.step(j/30, *synthetic([0,1,2,3], frequency=f))
            c, _, force, _ = abrupt_model.step(j/30,
                dict(detected=j<8, center=[0.,0.]), u, q,
                allow_gain_learning=False, input_meta=info)
            abrupt.append(dict(frame=j, frequency_hz=f, active=force['active'],
                acceleration=c['acceleration']))
        expected = [0.,-200.*((speed/300.)**2-1.)]
        if speed == 360.:
            np.testing.assert_allclose(abrupt[-1]['acceleration'], expected, atol=1e-7)
        abrupt_cases[str(speed)] = dict(expected_proxy_change_px_s2=expected,
            observed_acceleration_px_s2=abrupt[-1]['acceleration'],
            response_preserved=speed == 360., rows=abrupt)
    return dict(partial_two_startup=True, visibility_4_3_2_4_no_false_acceleration=True,
        real_speed_response_px_s2=speed_response.tolist(), real_normal_response=True,
        missing_input_moved_to_unknown_immediately=True, reappearance_no_double_count=True,
        covariance_psd=True, pooled_is_change_only=True, causal_prefix=True, visibility_log=log,
        step_diagnostics=abrupt_cases,
        known_limit_harmonic_sized_step='No phase confirmation: qualification gap and reacquisition can absorb a real large speed change.')


def replay(frozen, arm):
    model = RotorForceObserver(); bank = LegacyRotorThrustBankProxy() if arm == 'stage161' else RotorThrustBankProxy()
    result = []; durations = []
    for old in frozen:
        started = time.perf_counter(); t = old['t']; s = old['structure']; g = old['geometry']
        current_pose = bool(len(g.get('rotors', [])) >= 2 or g.get('side', {}).get('valid'))
        frequency = old['period']['frequency_hz']
        u, q, meta = bank.step(t, s, g.get('rotors', []),
            old['rates'] if s.get('mode') in ('four','three','partial_two') else [],
            dict(qualified=bool(frequency and current_pose), frequency_hz=frequency))
        if not current_pose: u = None; q = 0.
        if arm == 'off': u = None; q = 0.; meta = {}
        c, valid, force, state = model.step(t, old['body'], u, q,
            allow_gain_learning=meta.get('learning_input_current', True),
            input_meta=None if arm == 'stage161' else meta)
        row = dict(t=t, body=old['body'], structure=s, coefficient=c, force=force,
            period_selection=meta, valid=bool(valid), state=state)
        row['physical'] = evaluate(row, [.1,.2,.4]) if c is not None else {}
        durations.append((time.perf_counter()-started)*1000); result.append(row)
    return result, dict(frames=len(result), valid=sum(r['valid'] for r in result),
        active_inputs=sum(r['force']['active'] for r in result),
        identified_inputs=sum(r['force']['active'] and r['period_selection'].get('source') == 'current_visible_slot_f2_sum' for r in result),
        mixed_change_inputs=sum(r['force']['active'] and r['period_selection'].get('mixed_spectrum',False) for r in result),
        partition_transfers=sum(r['force'].get('visibility_partition_changed',False) for r in result),
        state_and_bank_median_ms=float(np.median(durations)), state_and_bank_p95_ms=float(np.percentile(durations,95)))


def main(output):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    dump(out/'mechanism_checks.json', mechanism_checks())
    base = ROOT.parent/'output/rotor_observer_gpu_20261101'
    all_rows = {}; summary = {}
    for clip in ['clear', 'side']:
        frozen = [json.loads(line) for line in (base/f'{clip}_final/predictions.jsonl').read_text('utf-8').splitlines()]
        for arm in ['stage161', 'observed', 'off']:
            data, brief = replay(frozen, arm); all_rows[(clip,arm)] = data
            (out/f'{clip}_{arm}.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False,allow_nan=False) for r in data)+'\n','utf-8')
            summary[f'{clip}_{arm}'] = brief
        for a,b in zip(all_rows[(clip,'stage161')], frozen):
            assert a['valid'] == b['valid']
            np.testing.assert_allclose(a['coefficient']['position'],b['coefficient']['position'],rtol=0,atol=1e-9)
            np.testing.assert_allclose(a['force']['acceleration_px_s2'],b['force']['acceleration_px_s2'],rtol=0,atol=1e-9)
    # All inference is saved before the first reference label is opened.
    from compare_final_pipeline import reference_at, stats
    truth = sorted(json.loads((ROOT.parent/'output/dense_labels_side_110_1195/labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    queries = json.loads((base/'summary.json').read_text('utf-8'))['fixed_side_rows']
    scores = []; missing = {}
    for query in queries:
        q = query['query_t']; target = reference_at(truth, q+.4); item = dict(query_t=q)
        assert target is not None
        for arm in ['stage161','observed','off']:
            row = [r for r in all_rows[('side',arm)] if r['t'] <= q+1e-9][-1]
            if not row['valid'] or q-row['t'] > .16:
                item[arm] = None; missing[arm] = missing.get(arm,0)+1; continue
            h = float(q-row['t']+.4)
            item[arm] = float(np.linalg.norm(np.asarray(evaluate(row,[h])[str(h)])-target))
        scores.append(item)
    result = dict(runs=summary, scores={a:stats([r[a] for r in scores if r[a] is not None]) for a in ['stage161','observed','off']},
        scored_rows=scores, missing=missing, frozen_stage161_reproduced=True,
        labels_used_for_inference=False, no_front_end_rerun=True, new_end_to_end_fps_measured=False,
        metric_thrust_or_RPM_proven=False, source_time_only=True)
    snapshot = out/'source_snapshot'; snapshot.mkdir()
    hashes = {}
    for name in ['rotor_thrust_bank.py','rotor_force_observer.py','run_rotor_only.py','run_fast_structure.py',
            'rotor_cuda_device.py','rotor_cuda_worker.py','tools/probe_observed_force_partition.py']:
        data = (ROOT/name).read_bytes(); (snapshot/Path(name).name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    result['source_sha256'] = hashes
    dump(out/'summary.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output',required=True)
    main(p.parse_args().output)
