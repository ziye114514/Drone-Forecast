"""Stage 144: score pipeline predictions against the dense hand labels.

Labels are at 10 Hz in native pixels. For every labelled "present" frame at time
t the reference centre is the box centre; for each horizon h the prediction is
the one the pipeline published at t-h. Only anchors whose origin row exists are
scored, and every missing pair is reported instead of being dropped silently.

    python research/stage144/score_dense.py --labels output/dense_labels_side_110_1195/labels.json \
        --run default=output/stage143_e2e_plain/predictions.jsonl
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
sys.path[:0] = [str(ROOT / 'research/stage135')]
from predictor135 import forecast  # noqa: E402

HORIZONS = (.1, .2, .4)
ORIGIN_MATCH = .015


def load_labels(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    references = []
    for record in payload['records'].values():
        if record.get('state') != 'present' or not record.get('box'):
            continue
        box = record['box']
        references.append(dict(t=float(record['t']),
                               centre=[.5 * (box[0] + box[2]), .5 * (box[1] + box[3])],
                               box=[float(v) for v in box]))
    references.sort(key=lambda item: item['t'])
    return references, payload


def load_run(path):
    rows = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[round(float(row['t']), 6)] = row
    return rows


def nearest(rows, t, tolerance=ORIGIN_MATCH):
    best = None
    for stamp in rows:
        delta = abs(stamp - t)
        if best is None or delta < best[0]:
            best = (delta, stamp)
    if best is None or best[0] > tolerance:
        return None
    return rows[best[1]]


def score(references, rows, accel_scale=1.0, damping_alpha=0.0, use_emitted=False,
          compare_key=None, published_only=False):
    result = {h: [] for h in HORIZONS}
    missing = {h: 0 for h in HORIZONS}
    paired = {h: [] for h in HORIZONS}
    for reference in references:
        for horizon in HORIZONS:
            row = nearest(rows, reference['t'] - horizon)
            if row is None:
                missing[horizon] += 1
                continue
            if published_only and not row.get('valid',False):
                missing[horizon] += 1
                continue
            coefficient = row['coefficients'].get('optimized_geometry')
            if coefficient is None:
                missing[horizon] += 1
                continue
            emitted = ((row.get('predictions') or {}).get('optimized_geometry') or {}).get(str(horizon))
            if compare_key is not None:
                alternative = ((row.get('predictions') or {}).get(compare_key) or {}).get(str(horizon))
                if alternative is None:
                    missing[horizon] += 1
                    continue
                baseline_point = (np.asarray(emitted, float) if emitted is not None
                                  else np.asarray(forecast(coefficient, horizon), float))
                paired[horizon].append(dict(
                    t=reference['t'],
                    baseline_error=float(np.linalg.norm(baseline_point - np.asarray(reference['centre']))),
                    alternative_error=float(np.linalg.norm(np.asarray(alternative)
                                                           - np.asarray(reference['centre'])))))
                result[horizon].append(dict(
                    t=reference['t'],
                    error_px=float(np.linalg.norm(np.asarray(alternative) - np.asarray(reference['centre']))),
                    predicted=np.asarray(alternative).tolist(),
                    truth=reference['centre']))
                continue
            if use_emitted and emitted is not None:
                predicted = np.asarray(emitted, float)
                error = float(np.linalg.norm(predicted - np.asarray(reference['centre'])))
                result[horizon].append(dict(t=reference['t'], error_px=error,
                                            predicted=predicted.tolist(),
                                            truth=reference['centre']))
                continue
            if accel_scale != 1.0:
                coefficient = dict(coefficient,
                                   acceleration=(np.asarray(coefficient.get('acceleration', [0., 0.]))
                                                 * accel_scale).tolist())
            predicted = np.asarray(forecast(coefficient, horizon), float)
            if damping_alpha:
                origin = np.asarray(coefficient['position'], float)
                shrink = max(0.5, 1.0 - damping_alpha * horizon ** 2)
                predicted = origin + shrink * (predicted - origin)
            error = float(np.linalg.norm(predicted - np.asarray(reference['centre'])))
            result[horizon].append(dict(t=reference['t'], error_px=error,
                                        predicted=predicted.tolist(),
                                        truth=reference['centre']))
    table = {}
    for horizon, items in result.items():
        values = sorted(item['error_px'] for item in items)
        table[str(horizon)] = dict(
            n=len(values), missing=missing[horizon],
            mean=float(np.mean(values)) if values else None,
            median=float(np.median(values)) if values else None,
            p90=float(values[int(.9 * (len(values) - 1))]) if values else None)
    if compare_key is not None:
        for horizon, items in paired.items():
            if not items:
                continue
            base = float(np.mean([item['baseline_error'] for item in items]))
            alt = float(np.mean([item['alternative_error'] for item in items]))
            table[str(horizon)]['paired_baseline_mean'] = base
            table[str(horizon)]['paired_alternative_mean'] = alt
    return table, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--run', action='append', required=True, help='name=path to predictions.jsonl')
    parser.add_argument('--accel-scale', action='append', type=float, default=None,
                        help='score the run again with the 140 acceleration multiplied by this factor')
    parser.add_argument('--damping-alpha', type=float, default=0.0,
                        help='shrink the predicted displacement by k(h)=1-alpha*h^2 before scoring')
    parser.add_argument('--use-emitted', action='store_true',
                        help='score the predictions the run emitted instead of recomputing from coefficients')
    parser.add_argument('--compare-key', default=None,
                        help='score predictions[<key>] against the emitted baseline on the same anchors '
                             '(e.g. rotor_fused from the physical branches)')
    parser.add_argument('--published-only', action='store_true',
                        help='Require origin valid=true; excluded/unpublished origins count as missing. Legacy default keeps all available coefficients.')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    references, payload = load_labels(args.labels)
    scales = args.accel_scale if args.accel_scale else [1.0]
    report = dict(stage=144, labels=str(Path(args.labels).resolve()),
                  label_frames=len(payload['records']), present_references=len(references),
                  horizons=list(HORIZONS), damping_alpha=args.damping_alpha, runs={})
    report['published_only']=args.published_only
    for item in args.run:
        name, _, path = item.partition('=')
        rows = load_run(path)
        for scale in scales:
            table, detail = score(references, rows, accel_scale=scale,
                                  damping_alpha=args.damping_alpha,
                                  use_emitted=args.use_emitted,
                                  compare_key=args.compare_key,published_only=args.published_only)
            key = name if scale == 1.0 and len(scales) == 1 else f'{name}@accel{scale:g}'
            report['runs'][key] = dict(path=str(Path(path).resolve()), accel_scale=scale,
                                       table=table, detail=detail)
    out = Path(args.out) if args.out else (
        WORK / 'output/stage144_dense_eval/scores_dense.json')
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    summary = {name: {h: (None if value['mean'] is None else round(value['mean'], 2),
                          value['n'], value['missing'])
                      for h, value in run['table'].items()}
               for name, run in report['runs'].items()}
    print(json.dumps(dict(labels=report['labels'], references=len(references), summary=summary,
                          written=str(out)), ensure_ascii=False))


if __name__ == '__main__':
    main()
