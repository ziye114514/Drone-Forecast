"""Stage 143 sensitivity sweep for the robust-velocity rule.

The rule has two knobs (window length, disagreement threshold). A gain that only
exists at one knob setting is not a mechanism, so this prints the whole small grid
for both side windows plus the clear window (where it must stay a no-op).

    python research/stage143/sweep143.py
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
    continuous6=(WORK / 'output/stage141_periodic_axis_20261024/continuous6/predictions.jsonl',
                 'continuous6'),
    extension2=(WORK / 'output/stage140_visible_fallback_20261024/extension2/predictions.jsonl',
                'extension2'),
    natural6=(WORK / 'output/stage138_shared_objective_20261024/natural6/predictions.jsonl',
              'natural6'))


def evaluate(rows, pairs, model):
    outputs = {}
    for row in rows:
        outputs[round(row['t'], 6)] = model.step(row, {})
    errors = {}
    for pair in pairs:
        key = round(pair['origin'], 8)
        row = next((r for r in rows if round(r['t'], 8) == key), None)
        if row is None:
            continue
        baseline = row['coefficients'].get('optimized_geometry')
        corrected = outputs.get(key, {}).get('coefficient')
        if baseline is None or corrected is None:
            continue
        for name, coefficient in (('base', baseline), ('rule', corrected)):
            predicted = forecast(coefficient, pair['actual_h'])
            error = float(np.linalg.norm(np.asarray(predicted) - np.asarray(pair['truth_bbox_center'])))
            errors.setdefault((name, pair['h']), []).append(error)
    return {key: float(np.mean(value)) for key, value in errors.items()}, model


def main():
    scores = json.loads((WORK / 'output/stage128_RAW_independent_support_20261023/scores128.json'
                         ).read_text(encoding='utf-8-sig'))
    windows = (.20, .30, .40)
    disagreements = (100., 150., 250.)
    table = []
    for window in windows:
        for disagreement in disagreements:
            entry = dict(window=window, disagreement=disagreement, cases={})
            for name, (rows_path, pairs_case) in CASES.items():
                rows = [json.loads(line) for line in
                        Path(rows_path).read_text(encoding='utf-8').splitlines() if line.strip()]
                pairs = [p for p in scores['pairs'] if p['case'] == pairs_case]
                model = RobustVelocityModel(window=window, disagreement=disagreement)
                values, model = evaluate(rows, pairs, model)
                entry['cases'][name] = dict(
                    h04_base=values.get(('base', .4)), h04_rule=values.get(('rule', .4)),
                    h02_base=values.get(('base', .2)), h02_rule=values.get(('rule', .2)),
                    replaced=model.replaced,
                    delta_pct=(None if not values.get(('base', .4)) else
                               100 * (values[('rule', .4)] - values[('base', .4)]) / values[('base', .4)]))
            table.append(entry)
            print(json.dumps(entry, ensure_ascii=False))
    out = WORK / 'output/stage143_local_flicker_20261024/sweep143.json'
    out.write_text(json.dumps(dict(table=table), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(written=str(out))))


if __name__ == '__main__':
    main()
