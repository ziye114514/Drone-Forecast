"""Stage 12 analysis: where does a per-rotor blade-pass measurement stop?

Reads a stage12 run and summarises, per rotor id: how many measurement attempts it
received, how many were valid, which gate rejected the rest, how stable its
frequency was while it lived, and how long the id survived. Nothing is re-run and
no frozen file is touched.
"""
from pathlib import Path
import argparse
import collections
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in
            (run / 'observations.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]

    reasons = collections.Counter()
    per_id = {}
    tick_four = 0
    tick_valid_any = 0
    qualities = []
    snrs = []
    phases = []
    attempts = 0
    for row in rows:
        t = row['t']
        records = row['sensing']['rotors']
        if row['sensing']['four_frequencies']:
            tick_four += 1
        valid_here = 0
        for record in records:
            identifier = record['rotor_id']
            entry = per_id.setdefault(identifier, dict(first=t, last=t, attempts=0, valid=0,
                                                       frequencies=[], qualities=[], reasons=[]))
            entry['last'] = t
            diagnostics = record.get('frequency_diagnostics') or {}
            if not diagnostics:
                continue
            attempts += 1
            entry['attempts'] += 1
            if diagnostics.get('valid') and record.get('blade_pass_hz') is not None:
                entry['valid'] += 1
                valid_here += 1
                entry['frequencies'].append(float(record['blade_pass_hz']))
                entry['qualities'].append(float(record['frequency_quality'] or 0.))
                qualities.append(float(record['frequency_quality'] or 0.))
                snrs.append(float(diagnostics.get('peak_to_noise') or 0.))
                phases.append(float(diagnostics.get('phase_coherence') or 0.))
            else:
                reason = diagnostics.get('reason', 'unknown')
                reasons[reason] += 1
                entry['reasons'].append(reason)
        tick_valid_any += bool(valid_here)

    identities = []
    for identifier, entry in sorted(per_id.items()):
        frequencies = np.asarray(entry['frequencies'], float)
        stability = None
        if len(frequencies) >= 3 and frequencies.mean() > 0:
            stability = float(np.median(np.abs(np.diff(frequencies))) / frequencies.mean())
        identities.append(dict(
            rotor_id=identifier, first_t=entry['first'], last_t=entry['last'],
            life_s=entry['last'] - entry['first'], attempts=entry['attempts'], valid=entry['valid'],
            valid_fraction=entry['valid'] / max(entry['attempts'], 1),
            median_hz=None if not len(frequencies) else float(np.median(frequencies)),
            iqr_hz=None if len(frequencies) < 4 else float(np.percentile(frequencies, 75)
                                                           - np.percentile(frequencies, 25)),
            relative_step=None if stability is None else stability,
            top_reason=None if not entry['reasons'] else collections.Counter(entry['reasons']).most_common(1)[0][0]))

    report = dict(stage=12, run=str(run), raw=protocol['raw'], method=protocol['method'],
                  interval_s=[protocol['start_s'], protocol['start_s'] + protocol['duration_s']],
                  frontend=protocol['frontend'], frequency_layer=protocol['frequency_layer'],
                  processed_ticks=len(rows),
                  detected_ticks=sum(1 for row in rows if row['observation']['detected']),
                  ticks_with_any_valid_frequency=tick_valid_any,
                  ticks_with_all_four_frequencies=tick_four,
                  measurement_attempts=attempts,
                  valid_measurements=sum(1 for value in qualities),
                  rejection_reasons=dict(reasons),
                  quality=dict(n=len(qualities),
                               median=None if not qualities else float(np.median(qualities)),
                               p90=None if not qualities else float(np.percentile(qualities, 90))),
                  peak_to_noise=dict(median=None if not snrs else float(np.median(snrs)),
                                     p90=None if not snrs else float(np.percentile(snrs, 90))),
                  phase_coherence=dict(median=None if not phases else float(np.median(phases)),
                                       p90=None if not phases else float(np.percentile(phases, 90))),
                  identities=identities,
                  identities_over_1s=sum(1 for item in identities if item['life_s'] > 1.0),
                  interpretation='Diagnostic only. blade_pass_hz is a blade-passage proxy; mechanical '
                                 'RPM needs an independent blade count and harmonic check.')
    out = run / 'analysis.json'
    if out.exists():
        raise SystemExit('analysis.json already exists; refusing to overwrite')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(method=report['method'], ticks=report['processed_ticks'],
                          detected=report['detected_ticks'],
                          ticks_all_four=tick_four, ticks_any=tick_valid_any,
                          attempts=attempts, valid=report['valid_measurements'],
                          accepted_fraction=round(report['valid_measurements'] / max(attempts, 1), 3),
                          reasons=dict(reasons.most_common(8)),
                          quality_median=report['quality']['median'],
                          identities=len(identities),
                          identities_over_1s=report['identities_over_1s']), ensure_ascii=False))


if __name__ == '__main__':
    main()
