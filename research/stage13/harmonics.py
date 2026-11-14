"""Stage 13: half/double harmonic disambiguation for per-rotor blade-pass frequency.

Reads a stage12 run (frozen frontend + frozen frequency layer) and asks:

  Q1 within one tick, do the valid rotor frequencies of the same vehicle really
     split into a x2 family, or was the split an artefact of comparing per-id
     medians across the whole run?
  Q2 does a single identity stay in one family while it lives, or does it flip?
  Q3 if the split is real, how much of it does a causal fleet-consistency rule
     remove (every rotor of one quadrotor turns at a comparable rate)?

The rule is causal: at tick t it may only use the measurements of tick t (all of
which come from the same past 160 ms window), never a future value. Nothing in
src/, stage8..stage12 is modified.
"""
from pathlib import Path
import argparse
import collections
import json

import numpy as np

FLEET_TOLERANCE = 0.25   # relative distance to the fleet median that counts as "same family"
HARMONIC_RATIO = 1.6     # above this the measurement is treated as the double of the fleet rate
BLADE_COUNT = 2          # documented assumption of the frozen paper configuration


def load_rows(run):
    return [json.loads(line) for line in
            (run / 'observations.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]


def tick_measurements(row):
    out = []
    for record in row['sensing']['rotors']:
        diagnostics = record.get('frequency_diagnostics') or {}
        if not diagnostics.get('valid') or record.get('blade_pass_hz') is None:
            continue
        out.append(dict(rotor_id=record['rotor_id'], hz=float(record['blade_pass_hz']),
                        quality=float(record.get('frequency_quality') or 0.),
                        phase=float(diagnostics.get('phase_coherence') or 0.),
                        snr=float(diagnostics.get('peak_to_noise') or 0.),
                        ambiguous=bool(diagnostics.get('harmonic_ambiguous'))))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    rows = load_rows(run)

    multi_ticks = 0
    split_ticks = 0
    ratios_before = []
    ratios_after = []
    family_counts = collections.Counter()
    per_id_family = collections.defaultdict(list)
    corrections = 0
    corrections_by_id = collections.Counter()
    tick_records = []
    for row in rows:
        measurements = tick_measurements(row)
        if len(measurements) < 2:
            continue
        multi_ticks += 1
        values = np.array([item['hz'] for item in measurements], float)
        median = float(np.median(values))
        if median <= 0:
            continue
        relative = values / median
        spread_before = float(values.max() / values.min())
        ratios_before.append(spread_before)
        if spread_before > 1 + 2 * FLEET_TOLERANCE:
            split_ticks += 1
        corrected = []
        classes = []
        for item in measurements:
            ratio = item['hz'] / median
            if ratio >= HARMONIC_RATIO:
                corrected.append(item['hz'] / 2.)
                classes.append('double')
                corrections += 1
                corrections_by_id[item['rotor_id']] += 1
            elif ratio <= 1. / HARMONIC_RATIO:
                corrected.append(item['hz'] * 2.)
                classes.append('half')
                corrections += 1
                corrections_by_id[item['rotor_id']] += 1
            else:
                corrected.append(item['hz'])
                classes.append('base')
        family_counts.update(classes)
        for item, klass in zip(measurements, classes):
            per_id_family[item['rotor_id']].append((row['t'], klass))
        corrected = np.array(corrected, float)
        ratios_after.append(float(corrected.max() / corrected.min()))
        tick_records.append(dict(t=row['t'], median_hz=median,
                                 measured=[round(item['hz'], 1) for item in measurements],
                                 classes=classes,
                                 spread_before=round(spread_before, 3),
                                 spread_after=round(float(corrected.max() / corrected.min()), 3)))

    flips = {}
    for identifier, series in per_id_family.items():
        changes = 0
        previous = None
        for _t, klass in series:
            if previous is not None and klass != previous:
                changes += 1
            previous = klass
        flips[identifier] = dict(measurements=len(series), family_changes=changes,
                                 families=sorted({klass for _t, klass in series}))

    report = dict(stage=13, run=str(run), rule=dict(
        fleet_tolerance=FLEET_TOLERANCE, harmonic_ratio=HARMONIC_RATIO,
        blade_count_assumption=BLADE_COUNT,
        note='causal: only the current tick measurements are used'),
        ticks_with_two_or_more_valid=multi_ticks,
        ticks_with_family_split=split_ticks,
        split_fraction=split_ticks / max(multi_ticks, 1),
        spread_before=dict(median=float(np.median(ratios_before)) if ratios_before else None,
                           p90=float(np.percentile(ratios_before, 90)) if ratios_before else None),
        spread_after=dict(median=float(np.median(ratios_after)) if ratios_after else None,
                          p90=float(np.percentile(ratios_after, 90)) if ratios_after else None),
        family_counts=dict(family_counts),
        corrections=corrections,
        corrections_by_id=dict(corrections_by_id.most_common(12)),
        identity_family_stability=flips,
        identities_that_flip_family=sum(1 for value in flips.values() if value['family_changes'] > 0),
        sample_ticks=tick_records[:20],
        interpretation='A blade-pass measurement is a proxy for rotation rate; halving or doubling it '
                       'assumes the reported peak sat on a harmonic of the same rotor, which is only '
                       'supported by fleet consistency, not proven.')
    out = Path(args.out).resolve() if args.out else run / 'harmonics.json'
    if out.exists():
        raise SystemExit('refusing to overwrite an existing harmonics.json')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(ticks_multi=multi_ticks, split_ticks=split_ticks,
                          split_fraction=round(report['split_fraction'], 3),
                          spread_before_median=report['spread_before']['median'],
                          spread_after_median=report['spread_after']['median'],
                          families=report['family_counts'], corrections=corrections,
                          identities=len(flips),
                          identities_flipping=report['identities_that_flip_family'],
                          written=str(out)), ensure_ascii=False))


if __name__ == '__main__':
    main()
