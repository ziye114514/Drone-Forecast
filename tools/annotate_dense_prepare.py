"""Render dense event frames for continuous hand labelling.

One sequential pass over the RAW (no seeking), fixed sample interval, one image
per sample, plus a manifest the labelling GUI reads. Optional model overlay is
taken read-only from an existing stage10/stage12 JSONL by nearest timestamp;
no detection is re-run and no reconstructed intensity is used.

    python tools/annotate_dense_prepare.py --raw <RAW> --start 20 --duration 15 \
        --fps 20 --output output/dense_labels_train6_20_35 \
        --model-jsonl output/stage10_continuous_identity_20261017/strict10.jsonl
"""
from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from event_drone_forecast.raw_io import open_raw, raw_chunks  # noqa: E402

def normalize(counts):
    """The encoding used by the user's other project: clip(log1p(c)/max(log1p(c))*255)."""
    values = np.log1p(counts.astype(np.float32))
    high = float(values.max())
    return np.clip(values / high * 255 if high > 0 else values, 0, 255).astype(np.uint8)


def render(on, off, display):
    """ON warm / OFF cool on black, or the white display mode used elsewhere.

    A five-millisecond window is what makes this readable: over fifty milliseconds
    a moving rotor paints its whole disk and neighbouring structures merge.
    """
    on_image, off_image = normalize(on), normalize(off)
    if display == 'white':
        # event_display.py in the other project: [255-on, 255-on, 255-off]
        return np.dstack([255 - on_image, 255 - on_image, 255 - off_image]).astype(np.uint8)
    if display == 'onoff':
        return np.dstack([on_image, off_image, np.zeros_like(on_image)]).astype(np.uint8)
    raise ValueError(f'unknown display mode {display}')


def model_index(path):
    """Accept both cached schemas: stage10 (observation.bbox) and entry rows (body.bbox)."""
    if path is None:
        return []
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if 'body' in row:
            body = row['body']
            detected = bool(body.get('detected'))
            box = body.get('bbox') if detected else None
        else:
            observation = row.get('observation', {})
            detected = bool(observation.get('detected'))
            box = observation.get('bbox') if detected else None
        rows.append((float(row['t']), detected, box))
    rows.sort(key=lambda item: item[0])
    return rows


def nearest(rows, t):
    if not rows:
        return None
    best = min(rows, key=lambda item: abs(item[0] - t))
    return best if abs(best[0] - t) <= 0.03 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--start', type=float, default=20.)
    parser.add_argument('--duration', type=float, default=15.)
    parser.add_argument('--fps', type=float, default=20.)
    parser.add_argument('--window-us', type=int, default=5000,
                        help='accumulation window inside each sample instant (the other project '
                             'uses 5000 us; longer windows smear a moving target)')
    parser.add_argument('--display', default='white', choices=['white', 'onoff'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-jsonl', default=None)
    parser.add_argument('--note', default='')
    args = parser.parse_args()

    step_us = int(round(1e6 / args.fps))
    folder = Path(args.output).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit(f'{folder} already exists and is not empty; choose a new directory')
    frames_dir = folder / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows = model_index(args.model_jsonl)
    reader = open_raw(args.raw)
    height, width = reader.get_size()
    frames = []
    for index, (stamp, events) in enumerate(
            raw_chunks(reader, args.start, args.duration, step_us)):
        # Only the last window_us of this sample interval is integrated: the sample
        # instant is the end of the window, as in the other project's ON/OFF frames.
        end_us = int(round(stamp * 1e6))
        window = events[events['t'] >= end_us - args.window_us] if len(events) else events
        pixel = (window['y'].astype(np.int32) * width + window['x']) if len(window) else np.empty(0, int)
        total = np.bincount(pixel, minlength=width * height).reshape(height, width)
        positive = np.bincount(pixel[window['p'] > 0], minlength=width * height).reshape(height, width)
        image = render(positive, total - positive, args.display)
        name = f'frame_{index:06d}.png'
        (frames_dir / name).write_bytes(cv2.imencode('.png', image)[1].tobytes())
        overlay = nearest(rows, stamp)
        frames.append(dict(index=index, t=round(float(stamp), 4), image=f'frames/{name}',
                           events=int(len(window)),
                           model_detected=None if overlay is None else overlay[1],
                           model_bbox=None if overlay is None or overlay[2] is None
                           else [float(v) for v in overlay[2]]))
    del reader
    manifest = dict(raw=str(Path(args.raw).resolve()), start_s=args.start, duration_s=args.duration,
                    fps=args.fps, step_us=step_us, window_us=args.window_us,
                    sample_instant='end of the accumulation window',
                    width=width, height=height, display=args.display,
                    render=f'{args.window_us} us event counts, clip(log1p(c)/max(log1p(c))*255), '
                           f'display={args.display}',
                    coordinate_system='native_event_pixels',
                    model_overlay_source=None if args.model_jsonl is None else str(Path(args.model_jsonl).resolve()),
                    note=args.note, frame_count=len(frames), frames=frames)
    (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                          encoding='utf-8')
    print(json.dumps(dict(rendered=len(frames), width=width, height=height, fps=args.fps,
                          window_us=args.window_us, display=args.display, output=str(folder),
                          model_overlay_frames=sum(1 for f in frames if f['model_bbox'] is not None)),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
