"""Stage 143 study: compare bounded-correction variants on identical frozen rows.

Variants share the same rows, the same endpoints and the same scoring code:
  baseline140  cached 140 coefficient, unchanged (the number to beat: 48.776 px)
  global142    the 142 single-frequency pilot (harness check, 50.891 px)
  local143v2   stage143 local tracks projected on the 141 line normal
  tangent143   same tracks projected on the line DIRECTION (the new axis)
  veto143      no physical direction; poor local evidence only shrinks the 140 term

    python research/stage143/study143.py --case continuous6
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
sys.path[:0] = [str(ROOT / 'research/stage143'), str(ROOT / 'research/stage142'),
                str(ROOT / 'research/stage135')]
from model143 import Model as LocalModel  # noqa: E402
from variants143 import TangentModel, VetoModel, RobustVelocityModel  # noqa: E402
from feature_model142 import Model as GlobalModel  # noqa: E402
from predictor135 import forecast  # noqa: E402

CASES = {
    'continuous6': dict(rows=WORK / 'output/stage141_periodic_axis_20261024/continuous6/predictions.jsonl',
                        local=WORK / 'output/stage143_local_flicker_20261024/continuous6/ticks.jsonl',
                        pairs_case='continuous6'),
    'extension2': dict(rows=WORK / 'output/stage140_visible_fallback_20261024/extension2/predictions.jsonl',
                       local=None, pairs_case='extension2'),
    'natural6': dict(rows=WORK / 'output/stage138_shared_objective_20261024/natural6/predictions.jsonl',
                     local=None, pairs_case='natural6'),
    'train1': dict(rows=WORK / 'output/stage143_e2e_train1_plain/predictions.jsonl',
                   local=None, pairs_case='extension1'),
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='continuous6', choices=sorted(CASES))
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    case = CASES[args.case]
    rows = [json.loads(line) for line in
            Path(case['rows']).read_text(encoding='utf-8').splitlines() if line.strip()]
    local_rows = {}
    if case['local'] is not None and Path(case['local']).exists():
        for line in Path(case['local']).read_text(encoding='utf-8').splitlines():
            if line.strip():
                record = json.loads(line)
                local_rows[round(record['t'], 6)] = record

    models = dict(global142=GlobalModel(), local143v2=LocalModel(),
                  tangent143=TangentModel(), veto143=VetoModel(),
                  robustv143=RobustVelocityModel(),
                  robustv143_onesided=RobustVelocityModel(one_sided=True))
    outputs = {name: {} for name in models}
    for row in rows:
        key = round(row['t'], 6)
        local = local_rows.get(key, {})
        for name, model in models.items():
            outputs[name][key] = model.step(row) if name == 'global142' else model.step(row, local)

    scores = load_json(WORK / 'output/stage128_RAW_independent_support_20261023/scores128.json')
    pairs = [p for p in scores['pairs'] if p['case'] == case['pairs_case']]
    detail = {name: [] for name in ['baseline140', *models]}
    for pair in pairs:
        key = round(pair['origin'], 8)
        row = next((r for r in rows if round(r['t'], 8) == key), None)
        if row is None:
            continue
        baseline = row['coefficients'].get('optimized_geometry')
        if baseline is None:
            continue
        candidates = {'baseline140': baseline}
        for name in models:
            candidates[name] = outputs[name][key]['coefficient']
        for name, coefficient in candidates.items():
            if coefficient is None:
                continue
            predicted = forecast(coefficient, pair['actual_h'])
            error = float(np.linalg.norm(np.asarray(predicted) - np.asarray(pair['truth_bbox_center'])))
            detail[name].append(dict(h=pair['h'], origin=key, error_px=error,
                                     active=bool(outputs[name][key]['active'])
                                     if name in models else False))
    table = []
    for horizon in (.1, .2, .4):
        entry = dict(h=horizon)
        for name, items in detail.items():
            values = [item['error_px'] for item in items if item['h'] == horizon]
            entry[name] = None if not values else float(np.mean(values))
            entry[f'{name}_n'] = len(values)
            if name in models:
                entry[f'{name}_active'] = sum(item['active'] for item in items if item['h'] == horizon)
        table.append(entry)
    report = dict(stage=143, case=args.case, ticks=len(rows), pairs=len(pairs), table=table,
                  detail=detail,
                  gates={name: dict(active=getattr(model, 'active', None)) for name, model in models.items()},
                  interpretation='All variants share rows, endpoints and forecast code; the numbers are '
                                 'previously exposed development endpoints, not independent validation.')
    destination = Path(args.out) if args.out else (
        WORK / f'output/stage143_local_flicker_20261024/scores143_{args.case}.json')
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding='utf-8')
    print(json.dumps(dict(case=args.case, table=table), ensure_ascii=False))


if __name__ == '__main__':
    main()
