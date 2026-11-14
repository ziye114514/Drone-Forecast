"""Stage 143 error decomposition on the dense benchmark.

For every anchor the 0.4 s error has two parts:

  * how far the true path curves away from the straight line implied by the true
    instantaneous velocity at the anchor (a floor no velocity-based model can beat);
  * how much the model's own velocity differs from that instantaneous velocity.

The instantaneous velocity is a central difference of the observed track, so this
is a diagnostic and never a predictor. It answers whether the remaining error is
recoverable by a better velocity estimate or is dominated by curvature.
"""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
sys.path[:0] = [str(ROOT / 'research/stage135')]
from predictor135 import forecast  # noqa: E402

CASES = dict(
    continuous6=WORK / 'output/stage141_periodic_axis_20261024/continuous6/predictions.jsonl',
    extension2=WORK / 'output/stage140_visible_fallback_20261024/extension2/predictions.jsonl',
    train1=WORK / 'output/stage143_e2e_train1_plain/predictions.jsonl',
    natural6=WORK / 'output/stage138_shared_objective_20261024/natural6/predictions.jsonl',
)
DT = .02


def main():
    summary = {}
    for name, path in CASES.items():
        rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
                if line.strip()]
        observed = [(r['t'], np.array(r['body']['center'], float)) for r in rows
                    if r['body'].get('detected')]
        index = {round(t, 3): c for t, c in observed}
        stamps = sorted(index)
        errors, floors, velocity_errors = [], [], []
        for row in rows:
            coefficient = row['coefficients'].get('optimized_geometry')
            if coefficient is None or not row['body'].get('detected'):
                continue
            t = round(row['t'], 3)
            target = round(t + .4, 3)
            here, there = index.get(t), index.get(target)
            if here is None or there is None:
                continue
            position = np.array([stamp for stamp in stamps if abs(stamp - t) < 1e-6][0])
            i = stamps.index(position)
            if i == 0 or i + 1 >= len(stamps):
                continue
            central = (index[stamps[i + 1]] - index[stamps[i - 1]]) / (2 * DT)
            floor = float(np.linalg.norm(there - here - central * .4))
            prediction = np.asarray(forecast(coefficient, .4), float)
            errors.append(float(np.linalg.norm(prediction - there)))
            floors.append(floor)
            model_velocity = np.asarray(coefficient.get('velocity', [0., 0.]), float)
            velocity_errors.append(float(np.linalg.norm(model_velocity - central)) * .4)
        if not errors:
            continue
        summary[name] = dict(
            n=len(errors),
            error_mean=float(np.mean(errors)), error_median=float(np.median(errors)),
            curvature_floor_mean=float(np.mean(floors)),
            curvature_floor_median=float(np.median(floors)),
            velocity_term_mean=float(np.mean(velocity_errors)),
            velocity_term_median=float(np.median(velocity_errors)),
            speed_median=float(np.median([np.linalg.norm(
                (index[stamps[i + 1]] - index[stamps[i - 1]]) / (2 * DT)) for i in range(1, len(stamps) - 1)])))
    out = WORK / 'output/stage143_local_flicker_20261024/decomposition.json'
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, item in summary.items():
        print(json.dumps(dict(case=name, **{k: round(v, 1) if isinstance(v, float) else v
                                            for k, v in item.items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
