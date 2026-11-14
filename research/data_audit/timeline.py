"""Read-only dump of sparse self-capture box-annotation timelines.

No RAW decode, no detector, no labels passed to any model. Only prints the
already-archived manual state/box timeline so a continuous passage can be
chosen before any inference is frozen.
"""
from pathlib import Path
import json
import sys

DATA = Path('E:/dataset')


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def timeline(record_id, folder):
    path = folder / 'annotations' / 'phase1_manual_boxes.json'
    if not path.exists():
        print(f'{record_id}: no annotation file')
        return
    payload = load(path)
    records = payload.get('records', payload)
    rows = []
    for key, value in records.items():
        if not isinstance(value, dict):
            continue
        event = value.get('event', {})
        box = value.get('box') or value.get('bbox') or event.get('box')
        stamp = event.get('timestamp_s', event.get('time_s', event.get('frame')))
        rows.append((stamp, key, event.get('state'), box))
    rows.sort(key=lambda r: (r[0] is None, r[0]))
    print(f'--- {record_id}: {len(rows)} records, file={path}')
    if rows:
        print('    record keys:', sorted(records[list(records)[0]].keys()))
        print('    event keys :', sorted(records[list(records)[0]].get('event', {}).keys()))
    runs = []
    for stamp, key, state, box in rows:
        has_box = box is not None
        if runs and runs[-1][0] == state and runs[-1][3] == has_box:
            runs[-1][2] = stamp
            runs[-1][4] += 1
        else:
            runs.append([state, stamp, stamp, has_box, 1])
    for state, begin, end, has_box, count in runs:
        print(f'    {state:>8}  box={str(has_box):>5}  {begin} -> {end}   n={count}')


if __name__ == '__main__':
    targets = sys.argv[1:] or ['1', '2', '3', '4', '5', '6']
    for name in targets:
        folder = DATA / 'train' / name
        if not folder.is_dir():
            folder = DATA / 'experiment' / name
        timeline(f'train/{name}' if (DATA / 'train' / name).is_dir() else f'experiment/{name}', folder)
