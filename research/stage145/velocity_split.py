"""Stage 145: separate the slow body translation from the fast box/rotor swing.

Item 2 of the plan asks to maintain body position/velocity apart from the rotor
disk geometry. In the side window there is no usable rotor structure, so the only
separable quantities are the slow component of the observed body centre and its
faster oscillation: if the box centre swings because the aircraft rotates (or the
box edges breathe), that swing must not become body velocity.

Every variant keeps the validated acceleration term and the 144 damping and only
replaces the velocity used for the extrapolation with a causal estimate over a
window: a plain linear fit (no filter memory) or a blend with the pipeline's own
filtered velocity.

    python research/stage145/velocity_split.py
"""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
sys.path[:0] = [str(ROOT/'core'),str(ROOT / 'research/stage135')]
from predictor135 import forecast  # noqa: E402

ALPHA = 0.9
HORIZONS = (.1, .2, .4)
LABELS = dict(
    side=(WORK / 'output/dense_labels_side_110_1195/labels.json',
          WORK / 'output/stage144_e2e_default_damped/predictions.jsonl'),
    train2=(WORK / 'output/dense_labels_train2_40_46/labels.json',
            WORK / 'output/stage144_e2e_train2_default/predictions.jsonl'),
)
VARIANTS = dict(
    pipeline=None,               # shipped velocity (filter state)
    fit15=('fit', 0.15), fit25=('fit', 0.25), fit40=('fit', 0.40),
    blend25=('blend', 0.25), blend40=('blend', 0.40),
)


def load_labels(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    points = []
    for record in payload['records'].values():
        if record.get('state') != 'present' or not record.get('box'):
            continue
        box = record['box']
        points.append((float(record['t']), np.array([.5 * (box[0] + box[2]), .5 * (box[1] + box[3])])))
    points.sort(key=lambda item: item[0])
    return points


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]
    centres = [(r['t'], np.array(r['body']['center'], float)) for r in rows if r['body'].get('detected')]
    return rows, centres


def slope(centres, t, window):
    recent = [(stamp, point) for stamp, point in centres if t - window <= stamp <= t + 1e-9]
    if len(recent) < 4:
        return None
    times = np.array([item[0] for item in recent])
    points = np.array([item[1] for item in recent])
    times = times - times.mean()
    design = np.column_stack([np.ones_like(times), times])
    solution, *_ = np.linalg.lstsq(design, points, rcond=None)
    return solution[1]


def evaluate(points, rows, centres, window, mode):
    errors = {h: [] for h in HORIZONS}
    for t, truth in points:
        for horizon in HORIZONS:
            row = next((r for r in rows if abs(r['t'] - (t - horizon)) <= .015), None)
            if row is None:
                continue
            coefficient = row['coefficients'].get('optimized_geometry')
            if coefficient is None:
                continue
            origin = np.asarray(coefficient['position'], float)
            acceleration = np.asarray(coefficient.get('acceleration', [0., 0.]), float)
            velocity = np.asarray(coefficient.get('velocity', [0., 0.]), float)
            if mode is not None:
                fitted = slope(centres, t - horizon, window)
                if fitted is None:
                    continue
                velocity = fitted if mode == 'fit' else .5 * velocity + .5 * fitted
            predicted = origin + velocity * horizon + .5 * acceleration * horizon ** 2
            shrink = max(0.5, 1.0 - ALPHA * horizon ** 2)
            predicted = origin + shrink * (predicted - origin)
            errors[horizon].append(float(np.linalg.norm(predicted - truth)))
    return {h: (float(np.mean(v)) if v else None, len(v)) for h, v in errors.items()}


def main():
    report = {}
    for case, (label_path, run_path) in LABELS.items():
        points = load_labels(label_path)
        rows, centres = load_rows(run_path)
        report[case] = {}
        for name, setting in VARIANTS.items():
            mode, window = (None, None) if setting is None else setting
            values = evaluate(points, rows, centres, window, mode)
            report[case][name] = values
            print(case, name, {h: (None if v[0] is None else round(v[0], 2), v[1])
                               for h, v in values.items()}, flush=True)
    out = WORK / 'output/stage145_velocity_split/results.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(written=str(out), note='fit<W>/blend<W> use a causal window of W seconds')))


if __name__ == '__main__':
    main()
