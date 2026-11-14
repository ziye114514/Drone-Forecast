"""Stage 143 dense benchmark: score every tick against the future observed centre.

The sparse manual boxes give only 10 endpoints on the side window, which cannot
separate a mechanism from a coincidence. This benchmark uses the pipeline's own
observed body centre at t+h as the reference for every tick that has one, so the
sample is the whole window (hundreds of anchors). It measures extrapolation of the
observed track, not absolute accuracy, and it is honest to say so in the report.

    python research/stage143/dense_score.py --case continuous6
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
sys.path[:0] = [str(ROOT / 'research/stage143'), str(ROOT / 'research/stage135')]
from variants143 import RobustVelocityModel  # noqa: E402
from predictor135 import forecast  # noqa: E402

CASES = dict(
    continuous6=WORK / 'output/stage141_periodic_axis_20261024/continuous6/predictions.jsonl',
    extension2=WORK / 'output/stage140_visible_fallback_20261024/extension2/predictions.jsonl',
    train1=WORK / 'output/stage143_e2e_train1_plain/predictions.jsonl',
    natural6=WORK / 'output/stage138_shared_objective_20261024/natural6/predictions.jsonl',
)
HORIZONS = (.1, .2, .4)
MATCH = .025


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='continuous6', choices=sorted(CASES))
    parser.add_argument('--accel-sweep', action='store_true',
                        help='scale the 140 acceleration term and report the dense error per scale')
    args = parser.parse_args()
    rows = [json.loads(line) for line in
            Path(CASES[args.case]).read_text(encoding='utf-8').splitlines() if line.strip()]
    observed = [(r['t'], np.array(r['body']['center'], float)) for r in rows
                if r['body'].get('detected')]

    def observed_at(target):
        best = None
        for stamp, centre in observed:
            delta = abs(stamp - target)
            if best is None or delta < best[0]:
                best = (delta, centre)
            elif stamp > target + MATCH:
                break
        return None if best is None or best[0] > MATCH else best[1]

    if args.accel_sweep:
        scales = (0.0, .25, .5, .75, 1.0, 1.25)
        errors = {scale: {h: [] for h in HORIZONS} for scale in scales}
        for row in rows:
            coefficient = row['coefficients'].get('optimized_geometry')
            if coefficient is None or not row['body'].get('detected'):
                continue
            for scale in scales:
                candidate = dict(coefficient,
                                 acceleration=(np.asarray(coefficient.get('acceleration', [0., 0.]))
                                               * scale).tolist())
                for horizon in HORIZONS:
                    reference = observed_at(row['t'] + horizon)
                    if reference is None:
                        continue
                    predicted = np.asarray(forecast(candidate, horizon), float)
                    errors[scale][horizon].append(float(np.linalg.norm(predicted - reference)))
        table = {str(scale): {str(h): (None if not v else dict(n=len(v), mean=float(np.mean(v))))
                              for h, v in per.items()} for scale, per in errors.items()}
        out = WORK / f'output/stage143_local_flicker_20261024/dense_accel_{args.case}.json'
        out.write_text(json.dumps(dict(case=args.case, table=table), ensure_ascii=False, indent=2),
                       encoding='utf-8')
        print(json.dumps(dict(case=args.case, means={s: {h: (None if v[h] is None else round(v[h]['mean'], 2))
                                                        for h in v}
                                                     for s, v in table.items()}),
                         ensure_ascii=False))
        return

    models = dict(baseline=None, robustv143=RobustVelocityModel(),
                  onesided=RobustVelocityModel(one_sided=True))
    errors = {name: {h: [] for h in HORIZONS} for name in models}
    for row in rows:
        coefficient = row['coefficients'].get('optimized_geometry')
        if coefficient is None or not row['body'].get('detected'):
            continue
        for name, model in models.items():
            candidate = coefficient
            if model is not None:
                candidate = model.step(row, {}).get('coefficient')
            if candidate is None:
                continue
            for horizon in HORIZONS:
                reference = observed_at(row['t'] + horizon)
                if reference is None:
                    continue
                predicted = np.asarray(forecast(candidate, horizon), float)
                errors[name][horizon].append(float(np.linalg.norm(predicted - reference)))
    report = dict(case=args.case, ticks=len(rows), reference='future observed body centre',
                  match_window_s=MATCH, table={})
    for name, per_horizon in errors.items():
        report['table'][name] = {}
        for horizon, values in per_horizon.items():
            if not values:
                continue
            array = sorted(values)
            report['table'][name][str(horizon)] = dict(
                n=len(array), median=float(np.median(array)), mean=float(np.mean(array)),
                p90=float(array[int(.9 * (len(array) - 1))]))
    out = WORK / f'output/stage143_local_flicker_20261024/dense_{args.case}.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(case=args.case, table={k: {h: (None if v.get('n', 0) == 0 else
                                                     round(v['mean'], 2))
                                                    for h, v in t.items()}
                                              for k, t in report['table'].items()},
                          n={h: report['table']['baseline'].get(str(h), {}).get('n')
                             for h in HORIZONS}), ensure_ascii=False))


if __name__ == '__main__':
    main()
