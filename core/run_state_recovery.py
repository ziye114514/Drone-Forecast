"""Experimental RAW-only ON/OFF rotor observations with protected tracking.

This entry point produces current image measurements, not future positions.
No G/intensity reconstruction, labels, learned detector, or physical truth is
used. Output contains every processed 10 ms tick, including recovery audits.
An explicit --start starts a cold tracker there; no hidden warmup is added.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'core'),str(ROOT / 'src'), str(ROOT / 'research/stage9')]
from recovery import RecoveryFrontend, PARAMS
from event_drone_forecast.raw_io import open_raw, raw_chunks

STEP_US = 10000
PROTECTED_ROOT = Path('D:/yolov10-main').resolve()


def json_text(value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=convert)


def fingerprints():
    paths = [Path(__file__), ROOT / 'research/stage9/recovery.py',
             ROOT / 'research/stage8/polarity.py',
             *(ROOT / 'src/event_drone_forecast').glob('*.py')]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def preview_frame(total, observation, audit):
    # Fixed scaling of this batch's event COUNTS, not reconstructed intensity.
    gray = np.uint8(np.clip(np.log1p(total) / np.log(21.), 0, 1) * 255)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for rotor in observation['rotors']:
        cv2.ellipse(frame,
                    ((rotor['cx'], rotor['cy']),
                     (rotor['major'], rotor['minor']), rotor['angle']),
                    (0, 170, 255), 2)
    if observation['detected']:
        center = tuple(np.rint(observation['center']).astype(int))
        cv2.drawMarker(frame, center, (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
    lines = [f"t={observation['t']:.3f}s  {audit['status']}",
             '10ms native event COUNTS; not intensity reconstruction',
             'Current observations only; no future-position prediction']
    for j, line in enumerate(lines):
        cv2.putText(frame, line, (8, 22 + j * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 255, 0), 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--duration', type=float, default=30.)
    parser.add_argument('--method', choices=['strict10', 'neighbor10'], default='neighbor10')
    parser.add_argument('--preview', action='store_true', help='Save a native event-count overlay MP4.')
    args = parser.parse_args()
    if not math.isfinite(args.start) or args.start < 0:
        parser.error('--start must be a finite nonnegative time in seconds')
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error('--duration must be a finite positive duration in seconds')
    if round(args.duration * 1e6) < 1:
        parser.error('--duration must be at least one microsecond')
    raw_path = Path(args.raw).resolve()
    folder = Path(args.output).resolve()
    if raw_path.is_relative_to(PROTECTED_ROOT) or folder.is_relative_to(PROTECTED_ROOT):
        parser.error('This entry point cannot read or write the protected yolov10-main directory')
    if folder.exists():
        parser.error('Output directory already exists; choose a new directory')

    reader = open_raw(raw_path)
    height, width = reader.get_size()
    source_hashes = fingerprints()
    raw_stat = raw_path.stat()
    folder.mkdir(parents=True, exist_ok=False)
    manifest = dict(raw=str(raw_path), inference_inputs=[str(raw_path)],
                    raw_size_bytes=raw_stat.st_size, raw_mtime_ns=raw_stat.st_mtime_ns,
                    raw_full_file_hash_computed=False,
                    start_s=args.start, duration_s=args.duration,
                    method=args.method, step_us=STEP_US, output_every_processed_tick=True,
                    cold_start=True, hidden_warmup=False, width=width, height=height,
                    parameters=PARAMS, source_hashes=source_hashes,
                    G_used=False, labels_in_inference=False, YOLO_used=False,
                    current_observations_only=True, future_prediction_validated=False,
                    preview='Native 10ms event counts with fixed log scaling; no reconstructed intensity.'
                            if args.preview else None)
    (folder / 'run_manifest.json').write_text(json_text(manifest), encoding='utf-8')
    front = RecoveryFrontend(width, height, args.method)
    started = time.perf_counter()
    progress = started
    frames = events_seen = detected = transfers = rejected = 0
    model_times = []
    last_t = None
    video = None
    try:
        if args.preview:
            video = cv2.VideoWriter(str(folder / 'preview_event_counts.mp4'),
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    1e6 / STEP_US, (width, height))
            if not video.isOpened():
                raise RuntimeError('Could not open the optional MP4 preview writer')
        with (folder / 'observations.jsonl').open('x', encoding='utf-8') as stream:
            for stamp, events in raw_chunks(reader, args.start, args.duration, STEP_US):
                tick = time.perf_counter()
                observation, total = front.detect(events, stamp)
                model_times.append((time.perf_counter() - tick) * 1000.)
                stream.write(json_text(dict(t=stamp, observation=observation,
                                            audit=front.audit, candidates=front.candidates)) + '\n')
                frames += 1
                events_seen += len(events)
                detected += bool(observation['detected'])
                transfers += bool(front.audit.get('transferred'))
                rejected += bool(front.audit.get('state_rejection'))
                last_t = stamp
                if video is not None:
                    video.write(preview_frame(total, observation, front.audit))
                if time.perf_counter() - progress > 15:
                    print(json_text(dict(t=stamp, processed_ticks=frames, detected_ticks=detected)), flush=True)
                    progress = time.perf_counter()
    finally:
        if video is not None:
            video.release()
    current_hashes = fingerprints()
    if source_hashes != current_hashes:
        raise RuntimeError('Source files changed during this run; do not use it for comparison')
    summary = dict(processed_ticks=frames, detected_ticks=detected, events=events_seen,
                   first_interval_start_s=args.start, last_tick_s=last_t,
                   confirmed_transfer_events=transfers, state_rejection_events=rejected,
                   wall_s=time.perf_counter() - started,
                   detect_ms_median=float(np.median(model_times)) if model_times else None,
                   detect_ms_p95=float(np.percentile(model_times, 95)) if model_times else None,
                   source_unchanged=True, source_hashes=source_hashes,
                   interpretation='Current image-plane rotor observations only. No extrapolated detections, '
                                  'calibrated attitude/RPM truth, or verified future-position prediction.')
    (folder / 'summary.json').write_text(json_text(summary), encoding='utf-8')
    print(json_text(summary), flush=True)


if __name__ == '__main__':
    main()
