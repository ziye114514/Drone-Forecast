"""Compare event accumulation windows for hand labelling, against the encoding
used by the user's other project (5 ms window, clip(log1p(count)/max*255)).

Read-only w.r.t. that project: this file only reproduces its documented encoding
so the labelling images look like the ones the user is used to. Prints sharpness
proxies, because a moving target smears as the window grows.

    python tools/render_compare.py --raw <RAW> --times 24.0,24.5,110.0
"""
from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from event_drone_forecast.raw_io import open_raw  # noqa: E402

WINDOWS_US = (5000, 10000, 20000, 50000)


def normalize_max(counts):
    """The other project's encoding: clip(log1p(c)/max(log1p(c))*255)."""
    values = np.log1p(counts.astype(np.float32))
    high = float(values.max())
    return np.clip(values / high * 255 if high > 0 else values, 0, 255).astype(np.uint8)


def normalize_percentile(counts, percentile=99.0):
    values = np.log1p(counts.astype(np.float32))
    positive = values[values > 0]
    high = float(np.percentile(positive, percentile)) if positive.size else 1.0
    return np.clip(values / max(high, 1e-6) * 255, 0, 255).astype(np.uint8)


def white_display(on, off):
    """Display mode 'white' from the other project: [255-on, 255-on, 255-off]."""
    return np.dstack([255 - on, 255 - on, 255 - off]).astype(np.uint8)


def metrics(on, off):
    total = on.astype(np.int32) + off.astype(np.int32)
    active = int((total > 0).sum())
    peak = float(np.percentile(total[total > 0], 95)) if active else 0.0
    gray = normalize_max(total).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return dict(events=int(total.sum()), active_pixels=active,
                events_per_active_pixel=round(float(total.sum()) / max(active, 1), 2),
                p95_count=peak, gradient_energy=round(float(np.mean(np.hypot(gx, gy))), 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--times', default='24.0')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    times = [float(item) for item in args.times.split(',') if item.strip()]
    reader = open_raw(args.raw)
    height, width = reader.get_size()
    report = []
    for t in sorted(times):
        # The SDK reader cannot seek backward, so load the longest window once and
        # derive every shorter window from the same events (no extra seeking).
        lo = int(round(t * 1e6))
        reader.seek_time(lo)
        events = reader.load_delta_t(max(WINDOWS_US))
        events = events[(events['t'] >= lo) & (events['t'] < lo + max(WINDOWS_US))]
        entry = dict(t=t, windows={})
        for window in WINDOWS_US:
            chunk = events[events['t'] < lo + window]
            pixel = (chunk['y'].astype(np.int32) * width + chunk['x']) if len(chunk) else np.empty(0, int)
            total = np.bincount(pixel, minlength=width * height).reshape(height, width)
            on = np.bincount(pixel[chunk['p'] > 0], minlength=width * height).reshape(height, width)
            off = total - on
            entry['windows'][window] = metrics(on, off)
            if args.output and window in (5000, 50000):
                folder = Path(args.output).resolve()
                folder.mkdir(parents=True, exist_ok=True)
                for name, image in (
                        ('white', white_display(normalize_max(on), normalize_max(off))),
                        ('max', np.dstack([normalize_max(on), normalize_max(off),
                                           np.zeros((height, width), np.uint8)]).astype(np.uint8))):
                    path = folder / f't{t:.2f}_{window // 1000}ms_{name}.png'
                    path.write_bytes(cv2.imencode('.png', image)[1].tobytes())
        report.append(entry)
    del reader
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
