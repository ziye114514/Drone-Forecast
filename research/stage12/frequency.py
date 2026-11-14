"""Stage 12: attach the per-ID frequency layer to the stage9 ON/OFF frontend.

The stage8/9/10 lineage never instantiated `RotorSensing`, and `PaperPerception`
deliberately returns all-NaN frequency, so those runs have no frequency layer at
all: `blade_pass_hz` is structurally null. This entry point wires the frozen
`RotorSensing` (per-rotor-ID frequency owner, unmodified) on top of the frozen
stage9 `RecoveryFrontend` and records, per tick, where each rotor's blade-pass
measurement stops.

Nothing in src/, stage8 or stage9 is modified; this file only instantiates them.
The protocol is written before the RAW is decoded and the output directory is
never overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
from collections import deque

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'core'),str(ROOT / 'src'), str(ROOT / 'research/stage9')]
from recovery import RecoveryFrontend, PARAMS  # noqa: E402
from event_drone_forecast.raw_io import open_raw, raw_chunks  # noqa: E402
from event_drone_forecast.paper_config import PaperConfig  # noqa: E402
from event_drone_forecast.rotor_sensing import RotorSensing  # noqa: E402

STEP_US = 10000
PROTECTED_ROOT = Path('D:/yolov10-main').resolve()

QUESTIONS = dict(
    Q1='does a rotor id ever reach a valid blade-pass measurement on this frontend',
    Q2='if not, which gate rejects it (reason histogram of periodic_measurement)',
    Q3='how many frequency updates does an id receive before it expires')


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
             ROOT / 'src/event_drone_forecast/rotor_sensing.py',
             ROOT / 'src/event_drone_forecast/paper_config.py',
             *(ROOT / 'src/event_drone_forecast').glob('*.py')]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--start', type=float, default=108.)
    parser.add_argument('--duration', type=float, default=12.)
    parser.add_argument('--method', default='neighbor10', choices=['strict10', 'neighbor10'])
    args = parser.parse_args()

    raw_path = Path(args.raw).resolve()
    folder = Path(args.output).resolve()
    if raw_path.is_relative_to(PROTECTED_ROOT) or folder.is_relative_to(PROTECTED_ROOT):
        parser.error('This entry point cannot read or write the protected yolov10-main directory')
    if folder.exists():
        parser.error('Output directory already exists; choose a new directory')

    cfg = PaperConfig()
    source_hashes = fingerprints()
    stat = raw_path.stat()
    folder.mkdir(parents=True, exist_ok=False)
    protocol = dict(stage=12, raw=str(raw_path), raw_size_bytes=stat.st_size,
                    raw_mtime_ns=stat.st_mtime_ns, start_s=args.start, duration_s=args.duration,
                    step_us=STEP_US, method=args.method, cold_start=True, hidden_warmup=False,
                    frontend='frozen stage9 RecoveryFrontend',
                    frequency_layer='frozen RotorSensing + periodic_measurement, unmodified',
                    paper_config=dict(step_us=cfg.step_us, frequency_window_us=cfg.frequency_window_us,
                                      frequency_update_steps=cfg.frequency_update_steps,
                                      frequency_min_hz=cfg.frequency_min_hz,
                                      frequency_max_hz=cfg.frequency_max_hz,
                                      blade_count=cfg.blade_count,
                                      max_buffer_events=cfg.max_buffer_events,
                                      max_frequency_samples=cfg.max_frequency_samples,
                                      track_expiry_s=cfg.track_expiry_s),
                    stage9_parameters=PARAMS, questions=QUESTIONS,
                    source_hashes=source_hashes, created_before_decode=True)
    (folder / 'protocol.json').write_text(json_text(protocol), encoding='utf-8')

    reader = open_raw(raw_path)
    height, width = reader.get_size()
    front = RecoveryFrontend(width, height, args.method)
    sensing = RotorSensing(cfg)
    history = deque()
    began_run = time.perf_counter()
    progress = began_run
    rows = 0
    valid_rotor_measurements = 0
    frequencies_valid_ticks = 0
    detect_ms = []
    sense_ms = []
    reasons = {}
    with (folder / 'observations.jsonl').open('x', encoding='utf-8') as stream:
        for stamp, events in raw_chunks(reader, args.start, args.duration, STEP_US):
            began = time.perf_counter()
            observation, _total = front.detect(events, stamp)
            detect_ms.append((time.perf_counter() - began) * 1000.)
            # Same bounded thinning the original detector used, so the frequency
            # layer never sees more than its own buffer budget.
            stride = max(1, int(np.ceil(len(events) / max(1, cfg.max_buffer_events // 6))))
            history.append((stamp, events[::stride].copy()))
            while history and history[0][0] < stamp - cfg.frequency_window_us / 1e6 - .03:
                history.popleft()
            began = time.perf_counter()
            sensing_result = sensing.step(observation, history, stamp)
            sense_ms.append((time.perf_counter() - began) * 1000.)
            if sensing_result['four_frequencies']:
                frequencies_valid_ticks += 1
            for record in sensing_result['rotors']:
                diagnostics = record.get('frequency_diagnostics') or {}
                if record.get('frequency_valid'):
                    valid_rotor_measurements += 1
                elif diagnostics:
                    # A finished measurement has no 'reason' key: it was rejected by
                    # the validity gate (peak-to-noise or quality) instead.
                    reason = diagnostics.get('reason', 'below_validity_gate')
                    reasons[reason] = reasons.get(reason, 0) + 1
            stream.write(json_text(dict(t=stamp, event_count=int(len(events)),
                                        observation=observation, sensing=sensing_result)) + '\n')
            rows += 1
            if time.perf_counter() - progress > 20:
                print(json_text(dict(t=stamp, rows=rows, valid=valid_rotor_measurements)), flush=True)
                progress = time.perf_counter()
    del reader
    if source_hashes != fingerprints():
        raise RuntimeError('Source files changed during this run; do not use it for comparison')
    summary = dict(stage=12, raw=str(raw_path), start_s=args.start, duration_s=args.duration,
                   method=args.method, processed_ticks=rows,
                   valid_rotor_measurements=valid_rotor_measurements,
                   ticks_with_all_four_frequencies=frequencies_valid_ticks,
                   rejection_reasons=reasons,
                   detect_ms_median=float(np.median(detect_ms)) if detect_ms else None,
                   detect_ms_p95=float(np.percentile(detect_ms, 95)) if detect_ms else None,
                   sensing_ms_median=float(np.median(sense_ms)) if sense_ms else None,
                   sensing_ms_p95=float(np.percentile(sense_ms, 95)) if sense_ms else None,
                   wall_s=time.perf_counter() - began_run,
                   interpretation='Per-rotor blade-pass measurements from the frozen frequency layer. '
                                  'Mechanical RPM stays provisional without an independent blade-count '
                                  'and harmonic check.')
    (folder / 'summary.json').write_text(json_text(summary), encoding='utf-8')
    print(json_text(summary), flush=True)


if __name__ == '__main__':
    main()
