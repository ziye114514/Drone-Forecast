"""Stage 13: a side-view degradation policy, measured on a frozen stage10 run.

The policy is deliberately narrow. Per tick it decides between two publication
modes using only causal information:

  rotor mode  at least three measured rotors and the rigid association of the
              previous tick accepted them (stage11 tracker, orientation
              preserving). Position, extent, rotor set and direction are
              published as measured.
  body mode   anything else. The whole-body centre is still published, but the
              extent is frozen at the last rotor-mode value and direction is
              declared unavailable instead of being taken from a collapsed or
              inferred box.

This is the "side-view degradation" question: the stage6 silhouette axis was
already shown not to beat the historical CA baseline, so degradation here means
"keep whole-body position, stop claiming direction", not "switch to an axis".

Nothing in src/, stage8..stage12 is modified; the stage11 tracker is imported
read-only to obtain identity-consistent rotor sets.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'core'),str(ROOT / 'research/stage11')]
from identity import RigidTracker  # noqa: E402

FRAGMENT_FRACTION = 0.5
CENTER_JUMP_PX = 50.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='stage10 run directory')
    parser.add_argument('--method', default='neighbor10')
    args = parser.parse_args()
    run = Path(args.run).resolve()
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in
            (run / f'{args.method}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]

    tracker = RigidTracker()
    reference_extent = None
    frozen_extent = None
    raw_fragments = 0
    policy_fragments = 0
    raw_jumps = 0
    policy_jumps = 0
    previous_raw = None
    previous_policy = None
    modes = {'rotor': 0, 'body': 0}
    jumps_by_mode = {'rotor': 0, 'body': 0}
    published_by_mode = {'rotor': 0, 'body': 0}
    timeline = []
    for row in rows:
        state = row['state']
        observation = row['observation']
        centers = [[float(v) for v in center] for center in state['previous_rotors']]
        labels = tracker.labels(centers, row['t'])
        reference = state['reference']
        if reference is not None:
            reference_extent = reference['extent']
        measured_four = len(centers) == 4
        # Observable enough to claim a rotor-plane state: at least three measured
        # rotors and every one of them attached to an existing identity.
        rotor_ok = len(centers) >= 3 and all(label is not None for label in labels)
        if rotor_ok and measured_four:
            frozen_extent = float(state['last_extent'])
        mode = 'rotor' if rotor_ok else 'body'
        modes[mode] += 1
        published = bool(observation['detected'])
        raw_extent = float(state['last_extent'])
        policy_extent = raw_extent if mode == 'rotor' else (frozen_extent if frozen_extent else raw_extent)
        center = observation.get('center')
        if published and reference_extent:
            if raw_extent < FRAGMENT_FRACTION * reference_extent:
                raw_fragments += 1
            if policy_extent < FRAGMENT_FRACTION * reference_extent:
                policy_fragments += 1
        if published and center:
            published_by_mode[mode] += 1
            if previous_policy is not None and np.linalg.norm(np.asarray(center) - previous_policy) > CENTER_JUMP_PX:
                policy_jumps += 1
                jumps_by_mode[mode] += 1
            if previous_raw is not None and np.linalg.norm(np.asarray(center) - previous_raw) > CENTER_JUMP_PX:
                raw_jumps += 1
            previous_raw = np.asarray(center, float)
            previous_policy = np.asarray(center, float)
        timeline.append(dict(t=row['t'], mode=mode, published=published,
                             raw_extent=round(raw_extent, 1),
                             policy_extent=round(policy_extent, 1),
                             direction_claimed=mode == 'rotor'))

    runs = []
    current = None
    for item in timeline:
        if current is None or item['mode'] != current['mode']:
            current = dict(mode=item['mode'], start=item['t'], end=item['t'], ticks=1)
            runs.append(current)
        else:
            current['end'] = item['t']
            current['ticks'] += 1

    report = dict(stage=13, run=str(run), method=args.method, raw=protocol['raw'],
                  interval_s=[protocol['start_s'], protocol['start_s'] + protocol['duration_s']],
                  policy=dict(rotor_mode='>=3 measured rotors and rigid association accepted',
                              body_mode='otherwise: publish whole-body centre, freeze extent, '
                                        'declare direction unavailable',
                              fragment_definition=f'detected tick with whole extent < '
                                                  f'{FRAGMENT_FRACTION} x active reference extent',
                              centre_jump_definition=f'> {CENTER_JUMP_PX} px between consecutive '
                                                     f'published centres'),
                  ticks=len(rows), modes=modes,
                  raw_fragment_ticks=raw_fragments, policy_fragment_ticks=policy_fragments,
                  raw_centre_jumps=raw_jumps, policy_centre_jumps=policy_jumps,
                  published_by_mode=published_by_mode, centre_jumps_by_mode=jumps_by_mode,
                  mode_runs=[dict(mode=item['mode'], start=item['start'], end=item['end'],
                                  ticks=item['ticks']) for item in runs],
                  direction_claimed_ticks=modes['rotor'],
                  interpretation='Degradation policy measured on frozen observations. It cannot add '
                                 'detections; it changes what the system claims (extent and '
                                 'direction) during partial or unstable observation.')
    out = run / f'degradation_{args.method}.json'
    if out.exists():
        raise SystemExit('refusing to overwrite an existing degradation report')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(method=args.method, ticks=len(rows), modes=modes,
                          raw_fragments=raw_fragments, policy_fragments=policy_fragments,
                          raw_jumps=raw_jumps, policy_jumps=policy_jumps,
                          direction_claimed=modes['rotor'],
                          written=str(out)), ensure_ascii=False))


if __name__ == '__main__':
    main()
