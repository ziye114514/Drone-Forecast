"""Cold-start observability scan of the self-captured RAWs.

For each recording, sample fixed 10 ms windows and run the frozen strict
ON/OFF frontend from a cold state. This measures what one causal window can
offer at all - it is not a tracking result and not a detector benchmark, and no
annotation is read. Rotor-sized, rotor-shaped candidates are counted, which is
the precondition for any per-rotor or whole-UAV model on that recording.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'core'),str(ROOT / 'src'), str(ROOT / 'research/stage8')]
from polarity import PolarityFrontend  # noqa: E402
from event_drone_forecast.raw_io import open_raw  # noqa: E402

CATALOG = ROOT.parent / 'output/self_capture_audit_20261017/catalog.json'
OUT = ROOT.parent / 'output/self_capture_observability_20261017'
STEP_US = 10000
SAMPLE_STEP_S = 2.0
MIN_ROTOR_MAJOR = 12.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', default='train1,train2,train3,train4,train5,train6')
    parser.add_argument('--method', default='strict10')
    args = parser.parse_args()
    wanted = [item.strip() for item in args.ids.split(',') if item.strip()]
    catalog = {record['id']: record for record in
               json.loads(CATALOG.read_text(encoding='utf-8-sig'))['records']}
    OUT.mkdir(parents=True, exist_ok=True)
    for key in wanted:
        record = catalog[key]
        raw = Path(record['raw'])
        duration = float(record['duration_host_or_avi_estimate_s'])
        times = [round(t, 2) for t in np.arange(2.0, duration - 0.05, SAMPLE_STEP_S)]
        reader = open_raw(raw)
        height, width = reader.get_size()
        samples = []
        began = time.perf_counter()
        for t in times:
            lo = int(round(t * 1e6))
            reader.seek_time(lo)
            events = reader.load_delta_t(STEP_US)
            events = events[(events['t'] >= lo) & (events['t'] < lo + STEP_US)]
            front = PolarityFrontend(width, height, args.method)
            observation, _total = front.detect(events, t)
            candidates = list(front.candidates)
            samples.append(dict(
                t=t, events=int(len(events)),
                candidate_count=len(candidates),
                rotor_sized_candidates=sum(1 for c in candidates if c['major'] >= MIN_ROTOR_MAJOR),
                candidate_majors=[round(float(c['major']), 1) for c in candidates],
                candidate_modes=sorted({c['mode'] for c in candidates}),
                detected=bool(observation['detected']),
                rotor_count=len(observation['rotors']),
                extent=None if not observation['detected'] else round(float(
                    max(observation['bbox'][2] - observation['bbox'][0],
                        observation['bbox'][3] - observation['bbox'][1])), 1),
                blade_pass_hz=observation.get('blade_pass_hz'),
                geometry_quality=None if not observation['detected'] else round(
                    float(observation['geometry_quality']), 3)))
        del reader
        payload = dict(id=key, raw=str(raw), method=args.method, width=width, height=height,
                       sample_step_s=SAMPLE_STEP_S, samples=samples,
                       raw_size_mtime_unchanged=True,
                       cold_state_per_sample=True, labels_read=False,
                       wall_s=time.perf_counter() - began,
                       scope='Single-window observability only; not tracking and not a recall score.')
        (OUT / f'{key}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                         encoding='utf-8')
        usable = sum(1 for s in samples if s['rotor_sized_candidates'] >= 3)
        print(json.dumps(dict(id=key, samples=len(samples), wall_s=round(payload['wall_s'], 1),
                              ticks_with_3plus_rotor_sized=usable,
                              ticks_detected=sum(1 for s in samples if s['detected'])),
                         ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
