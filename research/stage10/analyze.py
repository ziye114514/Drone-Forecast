"""Post-hoc continuous identity / fragmentation / recapture audit.

Reads a stage10 run directory, applies the failure definitions frozen in its
protocol.json, and writes analysis.json plus a per-second timeline. This script
never touches the frozen stage8/stage9 code and never re-runs inference.

Rotor identity note: the frozen frontend keeps no persistent per-motor id.
`constellation` is rewritten in detection order whenever four rotors are found,
so identity only survives through the partial-constellation path. The audit
therefore measures whether a persistent id COULD be maintained: for every pair
of consecutive ticks with four measured rotors it finds the best rigid
(rotation + uniform scale + translation) correspondence. An odd permutation is
not realisable by a similarity on a planar point set, so an odd best
correspondence is a genuine exchange of two rotor labels, i.e. a rotor swap.
"""
from pathlib import Path
import argparse
import itertools
import json
import math
import sys

import numpy as np

EPS = 1e-9


def load_jsonl(path):
    rows = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def similarity_residual(a, b):
    """Best rotation+uniform scale+translation residual mapping a onto b."""
    za = a[:, 0] + 1j * a[:, 1]
    zb = b[:, 0] + 1j * b[:, 1]
    design = np.column_stack([za, np.ones_like(za)])
    solution, *_ = np.linalg.lstsq(design, zb, rcond=None)
    predicted = design @ solution
    return float(np.sqrt(np.mean(np.abs(predicted - zb) ** 2)))


def permutation_parity(order):
    order = list(order)
    seen = [False] * len(order)
    parity = 1
    for start in range(len(order)):
        if seen[start]:
            continue
        length = 0
        node = start
        while not seen[node]:
            seen[node] = True
            node = order[node]
            length += 1
        if length % 2 == 0:
            parity = -parity
    return parity


def pair_analysis(four_ticks, tol_floor=8.0, tol_fraction=0.06):
    """Consecutive four-rotor correspondence statistics."""
    events = []
    for previous, current in zip(four_ticks, four_ticks[1:]):
        a = np.asarray(previous['centers'], float)
        b = np.asarray(current['centers'], float)
        spacing = float(np.median([np.linalg.norm(a[i] - a[j])
                                   for i, j in itertools.combinations(range(4), 2)]))
        tol = max(tol_floor, tol_fraction * spacing)
        best = None
        for order in itertools.permutations(range(4)):
            residual = similarity_residual(a[list(order)], b)
            if best is None or residual < best[0]:
                best = (residual, order)
        identity = similarity_residual(a, b)
        residual, order = best
        events.append(dict(t=current['t'], best_residual=residual, identity_residual=identity,
                           tolerance=tol, spacing=spacing, order=list(order),
                           parity=permutation_parity(order),
                           rigid=bool(residual <= tol),
                           identity_competitive=bool(identity <= 1.5 * max(residual, EPS)),
                           spacing_change=float(spacing / max(previous['spacing'], EPS))))
    return events


def method_analysis(rows, definitions):
    timeline = []
    reference_extent = None
    recent_extents = []
    fragments = []
    transfers = []
    four_runs = []
    current_run = []
    detected_run = 0
    longest_detected_run = 0
    gap = 0
    longest_gap = 0
    rotor_histogram = {}
    for row in rows:
        state = row['state']
        observation = row['observation']
        audit = row['audit']
        reference = state['reference']
        if reference is not None:
            reference_extent = reference['extent']
        detected = bool(observation['detected'])
        extent = float(state['last_extent'])
        rotors = len(observation['rotors'])
        rotor_histogram[rotors] = rotor_histogram.get(rotors, 0) + 1
        timeline.append(dict(t=row['t'], detected=detected, rotor_count=rotors,
                             observed_rotor_count=int(observation.get('observed_rotor_count', 0)),
                             layout_inferred=bool(observation.get('layout_inferred')),
                             extent=extent, reference_extent=reference_extent,
                             candidates=len(row['candidates']),
                             challenge=None if state['challenge'] is None else state['challenge']['count'],
                             transferred=bool(audit.get('transferred')),
                             rejection=audit.get('state_rejection'),
                             status=audit.get('status'),
                             event_count=int(row['event_count'])))
        if detected:
            detected_run += 1
            longest_detected_run = max(longest_detected_run, detected_run)
            gap = 0
            recent_extents.append((row['t'], extent))
            recent_extents = [item for item in recent_extents if item[0] >= row['t'] - 1.0]
            baseline = reference_extent if reference_extent else (
                float(np.median([item[1] for item in recent_extents])) if recent_extents else None)
            if baseline and extent < 0.5 * baseline:
                fragments.append(dict(t=row['t'], extent=extent, baseline=baseline,
                                      ratio=extent / baseline))
        else:
            detected_run = 0
            gap += 1
            longest_gap = max(longest_gap, gap)
        if audit.get('transferred'):
            previous_center = None
            for earlier in reversed(timeline[:-1]):
                if earlier['detected']:
                    previous_center = earlier
                    break
            transfers.append(dict(t=row['t'],
                                  center=observation.get('center'),
                                  extent=extent,
                                  previous_detected_t=None if previous_center is None else previous_center['t'],
                                  previous_extent=None if previous_center is None else previous_center['extent']))
        if len(state['previous_rotors']) == 4:
            centers = [list(map(float, c)) for c in state['previous_rotors']]
            spacing = float(np.median([np.linalg.norm(np.asarray(centers[i]) - np.asarray(centers[j]))
                                       for i, j in itertools.combinations(range(4), 2)]))
            current_run.append(dict(t=row['t'], centers=centers, spacing=spacing))
        else:
            if len(current_run) > 1:
                four_runs.append(current_run)
            current_run = []
    if len(current_run) > 1:
        four_runs.append(current_run)
    pair_events = []
    for run in four_runs:
        pair_events.extend(pair_analysis(run))
    swaps = [event for event in pair_events if event['rigid'] and event['parity'] == -1]
    ambiguous = [event for event in pair_events if not event['rigid']]
    rigid = [event for event in pair_events if event['rigid']]
    return dict(
        ticks=len(rows),
        detected_ticks=sum(1 for item in timeline if item['detected']),
        coverage=sum(1 for item in timeline if item['detected']) / max(len(timeline), 1),
        rotor_count_histogram=rotor_histogram,
        four_rotor_ticks=rotor_histogram.get(4, 0),
        layout_inferred_ticks=sum(1 for item in timeline if item['layout_inferred']),
        longest_detected_run_ticks=longest_detected_run,
        longest_gap_ticks=longest_gap,
        fragments=fragments,
        transfers=transfers,
        four_rotor_runs=len(four_runs),
        four_rotor_pairs=len(pair_events),
        rigid_pairs=len(rigid),
        rigid_fraction=len(rigid) / max(len(pair_events), 1),
        rotor_swaps=swaps,
        ambiguous_pairs=len(ambiguous),
        identity_competitive_fraction=(sum(1 for event in rigid if event['identity_competitive'])
                                       / max(len(rigid), 1)),
        median_best_residual=float(np.median([event['best_residual'] for event in pair_events]))
        if pair_events else None,
        median_spacing_change=float(np.median([event['spacing_change'] for event in pair_events]))
        if pair_events else None,
        timeline=timeline)


def per_second(timeline):
    buckets = {}
    for item in timeline:
        key = int(math.floor(item['t']))
        bucket = buckets.setdefault(key, dict(n=0, detected=0, extents=[], rotors=[], transfers=0,
                                              fragments=0, candidates=[]))
        bucket['n'] += 1
        bucket['detected'] += bool(item['detected'])
        bucket['extents'].append(item['extent'])
        bucket['rotors'].append(item['rotor_count'])
        bucket['transfers'] += bool(item['transferred'])
        bucket['candidates'].append(item['candidates'])
    rows = []
    for key in sorted(buckets):
        bucket = buckets[key]
        rows.append(dict(second=key, ticks=bucket['n'],
                         detected_fraction=bucket['detected'] / bucket['n'],
                         median_extent=float(np.median(bucket['extents'])),
                         median_rotors=float(np.median(bucket['rotors'])),
                         median_candidates=float(np.median(bucket['candidates'])),
                         transfers=bucket['transfers']))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='stage10 output directory')
    args = parser.parse_args()
    folder = Path(args.run).resolve()
    protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    definitions = protocol['failure_definitions']
    report = dict(stage=10, run=str(folder), raw=protocol['raw'],
                  start_s=protocol['start_s'], duration_s=protocol['duration_s'],
                  failure_definitions=definitions, methods={})
    for method in protocol['methods']:
        rows = load_jsonl(folder / f'{method}.jsonl')
        analysis = method_analysis(rows, definitions)
        analysis['per_second'] = per_second(analysis.pop('timeline'))
        report['methods'][method] = analysis
        print(json.dumps(dict(method=method, ticks=analysis['ticks'],
                              coverage=round(analysis['coverage'], 4),
                              four_rotor_ticks=analysis['four_rotor_ticks'],
                              layout_inferred=analysis['layout_inferred_ticks'],
                              fragments=len(analysis['fragments']),
                              transfers=len(analysis['transfers']),
                              rigid_fraction=round(analysis['rigid_fraction'], 3),
                              rotor_swaps=len(analysis['rotor_swaps']),
                              ambiguous_pairs=analysis['ambiguous_pairs'],
                              longest_gap_ticks=analysis['longest_gap_ticks']), ensure_ascii=False))
    out = folder / 'analysis.json'
    if out.exists():
        raise SystemExit('analysis.json already exists; refusing to overwrite')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(written=str(out))), flush=True)


if __name__ == '__main__':
    main()
