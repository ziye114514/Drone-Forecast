"""Post-hoc sparse-label cross-check of a stage10 continuous run.

Labels are read ONLY here, after inference, and never enter the inference path.
The archived manual boxes live in the ds4 reconstruction frame (320x184 padded,
native 1280x720 / 4), so a box is scaled by 4 to native pixels. A manual box is
a coarse whole-UAV box drawn by the operator, not a rotor, centroid or attitude
truth, and the annotation cadence is ~0.67 s, so instants between annotations
are simply not judged.

  * published while labelled absent  -> false publication (the direct evidence
    for "recaptured a wrong target")
  * not published while labelled present -> miss
  * published while labelled present -> hit, with native-pixel centre error
"""
from pathlib import Path
import argparse
import json
import math

FPS = 40.0
BOX_SCALE = 4.0


def load_labels(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    rows = []
    for key, value in payload['records'].items():
        event = value.get('event', {})
        state = event.get('state')
        box = event.get('box')
        frame = int(value.get('avi_frame', key))
        rows.append(dict(t=(frame - 1) / FPS, state=state,
                         box=None if box is None else [float(v) * BOX_SCALE for v in box],
                         frame=frame))
    rows.sort(key=lambda item: item['t'])
    return rows


def nearest(rows, t, tolerance):
    lo, hi = 0, len(rows) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        delta = rows[mid]['t'] - t
        if best is None or abs(delta) < abs(best[1]):
            best = (rows[mid], delta)
        if delta < 0:
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None or abs(best[1]) > tolerance:
        return None
    return best[0]


def episodes(times, gap=1.0):
    out = []
    for t in times:
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
            out[-1][2] += 1
        else:
            out.append([t, t, 1])
    return [dict(start=round(a, 2), end=round(b, 2), instants=n) for a, b, n in out]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--tolerance', type=float, default=0.35)
    args = parser.parse_args()
    folder = Path(args.run).resolve()
    protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    labels = load_labels(args.labels)
    report = dict(run=str(folder), labels=str(Path(args.labels).resolve()),
                  tolerance_s=args.tolerance, fps=FPS, box_scale=BOX_SCALE,
                  label_states={}, methods={},
                  scope='Post-hoc sparse cross-check only; labels were not used in inference. '
                        'Manual boxes are coarse whole-UAV boxes in ds4 x4 pixels; instants between '
                        'annotations (about 0.67 s apart) are not judged.')
    for state in sorted({row['state'] for row in labels}):
        report['label_states'][state] = sum(1 for row in labels if row['state'] == state)
    for method in protocol['methods']:
        rows = [json.loads(line) for line in
                (folder / f'{method}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        ticks = dict(present_published=0, present_missing=0, absent_published=0, absent_quiet=0,
                     unjudged_published=0)
        instant = {}
        centre_errors = []
        for row in rows:
            label = nearest(labels, row['t'], args.tolerance)
            if label is None:
                ticks['unjudged_published'] += bool(row['observation']['detected'])
                continue
            published = bool(row['observation']['detected'])
            slot = instant.setdefault(label['frame'], dict(state=label['state'], published=False,
                                                           box=label['box']))
            slot['published'] = slot['published'] or published
            if label['state'] == 'present':
                if published:
                    ticks['present_published'] += 1
                    if label['box'] is not None and row['observation'].get('center'):
                        cx, cy = row['observation']['center']
                        bx = 0.5 * (label['box'][0] + label['box'][2])
                        by = 0.5 * (label['box'][1] + label['box'][3])
                        centre_errors.append(math.hypot(cx - bx, cy - by))
                else:
                    ticks['present_missing'] += 1
            elif label['state'] == 'absent':
                ticks['absent_published' if published else 'absent_quiet'] += 1
            else:
                ticks['unjudged_published'] += published
        present_instants = [slot for slot in instant.values() if slot['state'] == 'present']
        absent_instants = [slot for slot in instant.values() if slot['state'] == 'absent']
        hit_instants = [slot for slot in present_instants if slot['published']]
        false_instants = [slot for slot in absent_instants if slot['published']]
        miss_times = [frame / FPS for frame, slot in instant.items()
                      if slot['state'] == 'present' and not slot['published']]
        false_times = [frame / FPS for frame, slot in instant.items()
                       if slot['state'] == 'absent' and slot['published']]
        centre_errors.sort()
        report['methods'][method] = dict(
            judged_present_instants=len(present_instants),
            hit_instants=len(hit_instants),
            miss_instants=len(present_instants) - len(hit_instants),
            instant_hit_rate=len(hit_instants) / max(len(present_instants), 1),
            labelled_absent_instants=len(absent_instants),
            absent_instants_with_publication=len(false_instants),
            absent_instant_publication_rate=len(false_instants) / max(len(absent_instants), 1),
            false_publication_episodes=episodes(false_times),
            miss_episodes=episodes(miss_times),
            tick_counts=ticks,
            centre_error_native_px=dict(
                n=len(centre_errors),
                median=None if not centre_errors else float(centre_errors[len(centre_errors) // 2]),
                p90=None if not centre_errors else float(centre_errors[int(0.9 * (len(centre_errors) - 1))]),
                max=None if not centre_errors else float(centre_errors[-1])),
            )
    out = folder / 'labels.json'
    if out.exists():
        raise SystemExit('labels.json already exists; refusing to overwrite')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for method, summary in report['methods'].items():
        print(json.dumps(dict(method=method,
                              present_instants=summary['judged_present_instants'],
                              hit_instants=summary['hit_instants'],
                              miss_instants=summary['miss_instants'],
                              hit_rate=round(summary['instant_hit_rate'], 3),
                              absent_instants=summary['labelled_absent_instants'],
                              absent_with_publication=summary['absent_instants_with_publication'],
                              false_episodes=len(summary['false_publication_episodes']),
                              median_centre_error_px=summary['centre_error_native_px']['median']),
                         ensure_ascii=False))
    print(json.dumps(dict(written=str(out))))


if __name__ == '__main__':
    main()
