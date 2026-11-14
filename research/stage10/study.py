"""Stage 10: continuous single-target identity/recovery audit on one self-captured RAW.

Frozen protocol, written before any inference:
  * one RAW, cold start at --start, --duration seconds, fixed 10 ms batches,
    no hidden warmup, no labels and no reconstruction in the inference path;
  * the same tick sequence is fed to every listed method;
  * extra JSONL columns are READ-ONLY snapshots of the frozen stage9 frontend
    state (constellation, reference, challenge). No stage8/stage9/src file is
    modified and no parameter is adapted after seeing results.

This entry writes observations only. Identity/fragmentation scoring is a
separate post-hoc script so the protocol stays fixed while the audit code
improves.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'core'),str(ROOT / 'src'), str(ROOT / 'research/stage9')]
from recovery import RecoveryFrontend, PARAMS  # noqa: E402
from event_drone_forecast.raw_io import open_raw, raw_chunks  # noqa: E402

STEP_US = 10000
METHODS = ('strict10', 'neighbor10')
METHOD_NOTE = dict(
    strict10='frozen stage8 strict native-pixel ON/OFF minimum, stage9 state protection',
    neighbor10='frozen stage8 3x3-neighbour ON/OFF minimum, stage9 state protection')
PROTECTED_ROOT = Path('D:/yolov10-main').resolve()

# Failure definitions are frozen here, before any RAW is decoded.
FAILURE_DEFINITIONS = dict(
    fragmentation='a detected tick whose whole extent < 0.5 x the active reference extent, '
                 'or, without a reference, < 0.5 x the median detected extent of the previous 1.0 s',
    rotor_swap='between two consecutive ticks with four measured rotors, the continuity assignment '
               'of rotor centres needs a similarity residual > max(8 px, 0.06 x median rotor spacing) '
               'while some alternative permutation fits below 0.5 x that residual',
    wrong_recapture='a confirmed transfer (audit.transferred) whose new centre is farther than '
                    'max(80 px, 1.3 x the whole extent at the last confirmed observation) from the '
                    'extrapolated previous centre, or any confirmed transfer inside a labelled-absent '
                    'stretch of the sparse annotation timeline',
    coverage='detected ticks / processed ticks over an identical tick sequence')


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


def vector(value):
    return None if value is None else [float(x) for x in np.asarray(value).ravel()]


def state_snapshot(front):
    """Read-only view of frozen frontend state; nothing is written back."""
    reference = None
    if front.reference is not None:
        reference = dict(center=vector(front.reference['center']),
                         extent=float(front.reference['extent']),
                         diameter=float(front.reference['diameter']),
                         t=float(front.reference['t']))
    challenge = None
    if front.challenge is not None:
        # _advance() merges the measurement keys into the challenge dict itself.
        challenge = dict(count=int(front.challenge['count']),
                         t=float(front.challenge['t']),
                         center=vector(front.challenge['center']),
                         extent=float(front.challenge['extent']),
                         diameter=float(front.challenge['diameter']))
    return dict(
        center=vector(front.center),
        velocity=vector(front.velocity),
        last_seen=None if front.last_seen is None else float(front.last_seen),
        last_t=None if front.last_t is None else float(front.last_t),
        last_extent=float(front.last_extent),
        constellation=[vector(r['center']) for r in (front.constellation or [])],
        constellation_t=None if front.constellation_t is None else float(front.constellation_t),
        previous_rotors=[vector(r['center']) for r in (front.previous_rotors or [])],
        layout_measured_t=None if front.layout_measured_t is None else float(front.layout_measured_t),
        reference=reference,
        challenge=challenge,
        normal=vector(front.normal),
        blade_pass_hz=None if not front.freqs else [
            None if not np.isfinite(f) else float(f) for f in front.freqs],
        frequency_quality=float(front.frequency_quality))


def run_method(raw_path, width, height, start, duration, method, folder):
    # A fresh reader per method: the SDK reader cannot seek backward, so every
    # method must decode the identical interval from its own cold reader.
    reader = open_raw(raw_path)
    front = RecoveryFrontend(width, height, method)
    ticks = detected = transfers = rejections = 0
    detect_ms = []
    rotors_seen = 0
    first_t = last_t = None
    with (folder / f'{method}.jsonl').open('x', encoding='utf-8') as stream:
        for stamp, events in raw_chunks(reader, start, duration, STEP_US):
            began = time.perf_counter()
            observation, _total = front.detect(events, stamp)
            detect_ms.append((time.perf_counter() - began) * 1000.)
            stream.write(json_text(dict(
                t=stamp,
                event_count=int(len(events)),
                observation=observation,
                audit=front.audit,
                candidates=front.candidates,
                state=state_snapshot(front))) + '\n')
            ticks += 1
            detected += bool(observation['detected'])
            transfers += bool(front.audit.get('transferred'))
            rejections += bool(front.audit.get('state_rejection'))
            rotors_seen += len(observation['rotors'])
            first_t = stamp if first_t is None else first_t
            last_t = stamp
    return dict(method=method, note=METHOD_NOTE[method], processed_ticks=ticks,
                detected_ticks=detected, coverage=detected / max(ticks, 1),
                confirmed_transfers=transfers, state_rejections=rejections,
                total_serialized_rotors=rotors_seen,
                first_tick_s=first_t, last_tick_s=last_t,
                detect_ms_median=float(np.median(detect_ms)) if detect_ms else None,
                detect_ms_p95=float(np.percentile(detect_ms, 95)) if detect_ms else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--duration', type=float, default=120.)
    parser.add_argument('--methods', default=','.join(METHODS))
    args = parser.parse_args()

    raw_path = Path(args.raw).resolve()
    folder = Path(args.output).resolve()
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    for method in methods:
        assert method in METHODS, method
    if raw_path.is_relative_to(PROTECTED_ROOT) or folder.is_relative_to(PROTECTED_ROOT):
        parser.error('This entry point cannot read or write the protected yolov10-main directory')
    if folder.exists():
        parser.error('Output directory already exists; choose a new directory')

    reader = open_raw(raw_path)
    height, width = reader.get_size()
    del reader
    source_hashes = fingerprints()
    stat = raw_path.stat()
    folder.mkdir(parents=True, exist_ok=False)
    protocol = dict(stage=10, question='continuous clear-to-side-to-clear identity and recovery audit',
                    raw=str(raw_path), raw_size_bytes=stat.st_size, raw_mtime_ns=stat.st_mtime_ns,
                    raw_full_file_hash_computed=False,
                    start_s=args.start, duration_s=args.duration, step_us=STEP_US,
                    cold_start=True, hidden_warmup=False, labels_in_inference=False,
                    reconstruction_used=False, yolo_used=False,
                    methods=methods, parameters=PARAMS,
                    failure_definitions=FAILURE_DEFINITIONS,
                    source_hashes=source_hashes,
                    created_before_decode=True, width=width, height=height)
    (folder / 'protocol.json').write_text(json_text(protocol), encoding='utf-8')
    started = time.perf_counter()
    summaries = []
    for method in methods:
        print(json_text(dict(starting=method, at_s=round(time.perf_counter() - started, 1))), flush=True)
        summaries.append(run_method(raw_path, width, height, args.start, args.duration, method, folder))
        print(json_text(summaries[-1]), flush=True)
    if source_hashes != fingerprints():
        raise RuntimeError('Source files changed during this run; do not use it for comparison')
    summary = dict(stage=10, raw=str(raw_path), start_s=args.start, duration_s=args.duration,
                   runs=summaries, wall_s=time.perf_counter() - started, source_unchanged=True,
                   labels_used_in_inference=False,
                   interpretation='Current image-plane observations only. Failure flags are computed '
                                  'post hoc against the frozen definitions in protocol.json. No '
                                  'calibrated attitude/RPM truth and no future-position accuracy.')
    (folder / 'summary.json').write_text(json_text(summary), encoding='utf-8')
    print(json_text(summary), flush=True)


if __name__ == '__main__':
    main()
