"""Score the burst-based blade detector against the archived sparse present/absent states.

The annotations are sparse (about 0.67 s apart) and only carry a state, not boxes,
so this scores *presence detection*, not localisation. A tick is judged only when
an annotation lies within --tolerance seconds of it.

    python research/stage16/score.py --run output/stage16_blade_detector_20261017/run_0_35 \
        --labels E:/dataset/train/6/annotations/phase1_manual_boxes.json
"""
from pathlib import Path
import argparse
import json
import statistics

FPS = 40.0


def load_states(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    rows = []
    for key, value in payload['records'].items():
        event = value.get('event', {})
        frame = int(value.get('avi_frame', key))
        rows.append(((frame - 1) / FPS, event.get('state')))
    rows.sort()
    return rows


def nearest(rows, t, tolerance):
    best = None
    for stamp, state in rows:
        delta = abs(stamp - t)
        if best is None or delta < best[0]:
            best = (delta, state)
        if stamp > t + tolerance:
            break
    if best is None or best[0] > tolerance:
        return None
    return best[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--tolerance', type=float, default=0.4)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    rows = load_states(args.labels)
    ticks = [json.loads(line) for line in
             (run / 'ticks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    buckets = {'present': dict(n=0, detected=0, regions=[], peaks=[]),
               'absent': dict(n=0, detected=0, regions=[], peaks=[])}
    per_second = {}
    false_ticks = []
    for tick in ticks:
        state = nearest(rows, tick['t'], args.tolerance)
        if state is None or state not in buckets:
            continue
        bucket = buckets[state]
        bucket['n'] += 1
        detected = bool(tick['regions'])
        bucket['detected'] += detected
        bucket['regions'].append(len(tick['regions']))
        bucket['peaks'].append(len(tick['peaks']))
        second = int(tick['t'])
        entry = per_second.setdefault(second, {'present': [0, 0], 'absent': [0, 0]})
        entry[state][0] += 1
        entry[state][1] += bool(tick['regions'])
        if state == 'absent' and detected:
            false_ticks.append(round(tick['t'], 2))
    report = dict(stage=16, run=str(run), labels=str(Path(args.labels).resolve()),
                  tolerance_s=args.tolerance, ticks=len(ticks), judged={}, per_second={},
                  false_positive_ticks=false_ticks[:60], false_positive_count=len(false_ticks))
    for state, bucket in buckets.items():
        report['judged'][state] = dict(
            ticks=bucket['n'], detected=bucket['detected'],
            detection_rate=bucket['detected'] / max(bucket['n'], 1),
            median_regions=statistics.median(bucket['regions']) if bucket['regions'] else None,
            median_peaks=statistics.median(bucket['peaks']) if bucket['peaks'] else None,
            ticks_with_four_peaks=sum(1 for value in bucket['peaks'] if value >= 4))
    for second in sorted(per_second):
        entry = per_second[second]
        report['per_second'][str(second)] = {
            state: (values[1] / values[0] if values[0] else None, values[0])
            for state, values in entry.items()}
    destination = Path(args.out).resolve() if args.out else run / 'score.json'
    if destination.exists():
        raise SystemExit('refusing to overwrite an existing score.json')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding='utf-8')
    print(json.dumps(dict(run=run.name, judged=report['judged'],
                          false_positive_count=report['false_positive_count'],
                          written=str(destination)), ensure_ascii=False))


if __name__ == '__main__':
    main()
