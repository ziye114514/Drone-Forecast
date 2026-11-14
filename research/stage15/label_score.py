"""Score motion models against dense hand labels (the ADE/FDE the sparse boxes cannot give).

Reference trajectory = the centre of each labelled "present" box; horizons are
interpolated between the two neighbouring labelled frames, so a prediction is
only scored when the labels bracket it. Models are the same causal filters used
in stage 14, imported read-only from research/stage14/motion.py.

    python research/stage15/label_score.py --run output/stage10_continuous_identity_20261017 \
        --method strict10 --labels output/dense_labels_train6_20_35/labels.json
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'core'),str(ROOT / 'research/stage14')]
from motion import build_filters, first_fix  # noqa: E402

SUB_STEP = 0.05          # ADE is integrated on this grid inside each horizon
REF_GAP = 0.08           # labels must bracket the target time by less than this


def load_reference(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    points = []
    for record in payload['records'].values():
        box = record.get('box')
        if record.get('state') != 'present' or not box:
            continue
        points.append((float(record['t']), 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])))
    points.sort(key=lambda item: item[0])
    return points, payload


def reference_at(points, t):
    if not points:
        return None
    times = [item[0] for item in points]
    if t < times[0] - REF_GAP or t > times[-1] + REF_GAP:
        return None
    index = int(np.searchsorted(times, t))
    if index == 0:
        return None if abs(times[0] - t) > REF_GAP else np.array(points[0][1:])
    if index >= len(points):
        return None if abs(times[-1] - t) > REF_GAP else np.array(points[-1][1:])
    before, after = points[index - 1], points[index]
    span = after[0] - before[0]
    if span > 0.25 or t - before[0] > REF_GAP or after[0] - t > REF_GAP:
        return None
    ratio = (t - before[0]) / span if span > 0 else 0.0
    return np.array(before[1:]) + ratio * (np.array(after[1:]) - np.array(before[1:]))


def load_series(path, method):
    source = Path(path) / f'{method}.jsonl'
    if not source.exists():
        source = Path(path) / 'observations.jsonl'
    series = []
    for line in source.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        observation = row['observation']
        center = observation.get('center') if observation['detected'] else None
        series.append((float(row['t']), None if center is None else (float(center[0]), float(center[1]))))
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--horizons', default='0.1,0.2,0.4')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    horizons = [float(item) for item in args.horizons.split(',') if item.strip()]

    points, payload = load_reference(args.labels)
    series = load_series(args.run, args.method)
    filters = build_filters()
    results = {model.name: {h: dict(ade=[], fde=[]) for h in horizons} for model in filters}
    anchors = {model.name: {h: 0 for h in horizons} for model in filters}

    last_t = None
    previous_fix = None
    pushed = 0
    for t, center in series:
        dt = None if last_t is None else max(t - last_t, 1e-3)
        velocity = None
        if center is not None and previous_fix is not None and dt:
            velocity = ((center[0] - previous_fix[0]) / dt, (center[1] - previous_fix[1]) / dt)
        for model in filters:
            if first_fix(model).x is None and center is not None:
                model.initialize(center, velocity)
                continue
            if first_fix(model).x is None:
                continue
            if dt is not None:
                model.predict(dt)
            if center is not None:
                model.update(center)
                pushed += 1
        if center is not None:
            previous_fix = center
        if pushed < 8:
            last_t = t
            continue
        for model in filters:
            for horizon in horizons:
                errors = []
                for step in np.arange(SUB_STEP, horizon + 1e-9, SUB_STEP):
                    reference = reference_at(points, t + float(step))
                    if reference is None:
                        errors = []
                        break
                    prediction = model.predict_at(float(step))
                    errors.append(float(np.hypot(prediction[0] - reference[0],
                                                 prediction[1] - reference[1])))
                if errors:
                    results[model.name][horizon]['ade'].append(float(np.mean(errors)))
                    results[model.name][horizon]['fde'].append(errors[-1])
                    anchors[model.name][horizon] += 1
        last_t = t

    summary = {}
    for model in filters:
        summary[model.name] = {}
        for horizon in horizons:
            ade = sorted(results[model.name][horizon]['ade'])
            fde = sorted(results[model.name][horizon]['fde'])
            summary[model.name][str(horizon)] = None if not ade else dict(
                n=len(ade), ade_median=float(np.median(ade)), ade_mean=float(np.mean(ade)),
                fde_median=float(np.median(fde)), fde_mean=float(np.mean(fde)))
    report = dict(stage=15, run=str(Path(args.run).resolve()), method=args.method,
                  labels=str(Path(args.labels).resolve()), label_frames=len(payload['records']),
                  present_reference_points=len(points), horizons=horizons, sub_step=SUB_STEP,
                  reference_gap_s=REF_GAP, models=summary,
                  note='Reference is the interpolated hand-label centre; a horizon is scored only '
                       'when labels bracket it. ADE averages the error on a 0.05 s grid, FDE is the '
                       'endpoint error.')
    destination = Path(args.out).resolve() if args.out else Path(args.run) / f'score_{args.method}.json'
    if destination.exists():
        raise SystemExit('refusing to overwrite an existing score report')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding='utf-8')
    print(json.dumps(dict(labels=report['labels'], present_points=len(points),
                          results={name: {h: (None if value is None else
                                              dict(n=value['n'], ade=round(value['ade_median'], 1),
                                                   fde=round(value['fde_median'], 1)))
                                          for h, value in horizons_.items()}
                                   for name, horizons_ in summary.items()},
                          written=str(destination)), ensure_ascii=False))


if __name__ == '__main__':
    main()
