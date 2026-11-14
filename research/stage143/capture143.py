"""Stage 143 capture: per-tick local flicker hypotheses on the side window.

The body ROI and body velocity are taken from the frozen 141 run (that run already
pays for the detector); the flicker pixels are recomputed from RAW with the
installed SDK engine, so no reconstructed frame and no cached pixel data is used.
Writes one JSONL record per 20 ms tick, observation only.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'core'),str(ROOT / 'src'), str(ROOT / 'research/stage141'), str(ROOT / 'research/stage143')]
from native_axis141 import NativeAxis  # noqa: E402
from regions143 import Tracker, cells, hypotheses, latest_per_pixel, roi_pixels, summarise  # noqa: E402
from event_drone_forecast.raw_io import open_raw, raw_chunks  # noqa: E402

CACHE = ROOT.parent / 'output/stage141_periodic_axis_20261024/continuous6/predictions.jsonl'
MIN_BODY_CONFIDENCE = .45


def json_text(value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=convert)


def load_cache():
    rows = {}
    for line in CACHE.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[round(float(row['t']), 6)] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--start', type=float, default=110.)
    parser.add_argument('--duration', type=float, default=9.5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    folder = Path(args.output).resolve()
    if folder.exists():
        raise SystemExit('output directory exists')
    folder.mkdir(parents=True)
    cache = load_cache()
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [Path(__file__), ROOT / 'research/stage143/regions143.py',
                         ROOT / 'research/stage141/native_axis141.py',
                         ROOT / 'research/stage79/native79.py']}
    protocol = dict(stage=143, raw=str(Path(args.raw).resolve()), start_s=args.start,
                    duration_s=args.duration, step_us=20000, sources=sources,
                    body_roi_source=str(CACHE.relative_to(ROOT.parent)),
                    pixels='recomputed from RAW with the installed SDK frequency engine (strict7)',
                    cell_px=24, cluster_log_tolerance=0.0392, track_log_gate=0.0953,
                    track_spatial_gate=90.0, observation_only=True,
                    note='Local flicker hypotheses for study; not RPM, thrust or attitude truth.')
    (folder / 'protocol.json').write_text(json_text(protocol), encoding='utf-8')

    reader = open_raw(args.raw)
    height, width = reader.get_size()
    observer = NativeAxis(width, height)
    tracker = Tracker()
    started = time.perf_counter()
    ticks = with_hypotheses = with_tracks = 0
    track_histogram = {}
    with (folder / 'ticks.jsonl').open('x', encoding='utf-8') as stream:
        for t, events in raw_chunks(reader, args.start, args.duration, 20000):
            began = time.perf_counter()
            q = observer.feed(events, t)
            row = cache.get(round(t, 6))
            body = row['body'] if row else None
            bbox = (body['bbox'] if body and body.get('detected')
                    and body.get('confidence', 0) >= MIN_BODY_CONFIDENCE else None)
            coefficient = (row or {}).get('coefficients', {}).get('matched_body') or {}
            velocity = coefficient.get('velocity', [0., 0.])
            x, y, frequency = roi_pixels(latest_per_pixel(q, width), bbox, t, velocity)
            cell_list = cells(x, y, frequency, bbox) if bbox is not None else []
            hyps = hypotheses(cell_list)
            live = tracker.step(hyps, t)
            summary = summarise(live, False)
            record = dict(t=round(t, 6), pixels=int(len(x)), cells=len(cell_list),
                          hypotheses=len(hyps), bbox=bbox, velocity=velocity,
                          global_hz=None if row is None else (row.get('periodic_axis') or {}).get('frequency_hz'),
                          **summary)
            stream.write(json_text(record) + '\n')
            ticks += 1
            with_hypotheses += bool(hyps)
            with_tracks += bool(summary.get('tracks'))
            track_histogram[summary.get('tracks', 0)] = track_histogram.get(summary.get('tracks', 0), 0) + 1
            if ticks % 100 == 0:
                print(json_text(dict(ticks=ticks, t=round(t, 2), hypotheses=len(hyps),
                                     tracks=summary.get('tracks'))), flush=True)
    del reader
    summary = dict(ticks=ticks, ticks_with_hypotheses=with_hypotheses, ticks_with_tracks=with_tracks,
                   track_count_histogram={str(k): v for k, v in sorted(track_histogram.items())},
                   wall_s=time.perf_counter() - started,
                   interpretation='Local flicker hypotheses only; a track is not a motor, four tracks are '
                                  'not four motors, and flicker frequency is not mechanical RPM.')
    (folder / 'summary.json').write_text(json_text(summary), encoding='utf-8')
    print(json_text(summary))


if __name__ == '__main__':
    main()
