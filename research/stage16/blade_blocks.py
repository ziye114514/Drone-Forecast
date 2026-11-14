"""Stage 16: burst-based blade detection in event-rate blocks (effqpdet idea, Python).

The current frontend needs ON and OFF events at the *same pixel* (min(on, off))
before it will even look at a blob. That gate is exactly what fails in a side
view, where each blade edge mostly produces one polarity. The SCIA 2025 detector
(effqpdet) instead treats a spatial block of events as a Poisson process and
looks for the periodic bursts a rotating blade leaves in the event rate. This
script reimplements that idea in numpy and measures the contrast between
segments where a drone is present and segments where it is absent.

For one time window: bin events into blocks of B x B pixels and 1 ms time bins,
take the periodogram of every block, and keep blocks whose strongest peak in the
blade-pass band stands well above the local median. Qualifying blocks are then
grouped into blade regions.

    python research/stage16/blade_blocks.py --raw <RAW>
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.signal import periodogram
from scipy.ndimage import label as ndlabel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from event_drone_forecast.raw_io import open_raw  # noqa: E402

BLOCK = 16                 # block edge in pixels
BIN_US = 1000              # time bin inside the window
WINDOW_US = 100_000        # analysis window
FMIN, FMAX = 60.0, 1600.0
MIN_EVENTS_PER_BLOCK = 60


def analyse(events, width, height, quality_thresholds, block):
    rows = height // block
    columns = width // block
    bins = WINDOW_US // BIN_US
    if not len(events):
        return None
    start = int(events['t'].min()) // BIN_US * BIN_US
    bin_index = ((events['t'].astype(np.int64) - start) // BIN_US).astype(np.int32)
    block_y = (events['y'].astype(np.int32) // block).astype(np.int32)
    block_x = (events['x'].astype(np.int32) // block).astype(np.int32)
    keep = (bin_index >= 0) & (bin_index < bins) & (block_y < rows) & (block_x < columns)
    flat_block = block_y[keep] * columns + block_x[keep]
    flat_bin = bin_index[keep]
    counts = np.bincount(flat_block * bins + flat_bin, minlength=rows * columns * bins)
    counts = counts.reshape(rows, columns, bins).astype(np.float32)
    totals = counts.sum(axis=2)
    series = counts.reshape(-1, bins)
    series = series - series.mean(axis=1, keepdims=True)
    frequencies, power = periodogram(series, fs=1e6 / BIN_US, window='hann', axis=1)
    band = (frequencies >= FMIN) & (frequencies <= FMAX)
    band_power = power[:, band]
    band_frequencies = frequencies[band]
    peak_index = np.argmax(band_power, axis=1)
    peak_power = band_power[np.arange(band_power.shape[0]), peak_index]
    median_power = np.median(band_power, axis=1)
    quality = peak_power / np.maximum(median_power, 1e-9)
    peak_hz = band_frequencies[peak_index]
    enough = totals.reshape(-1) >= MIN_EVENTS_PER_BLOCK
    result = dict(block=int(block), blocks=int(rows * columns), bins=int(bins),
                  enough_event_blocks=int(enough.sum()), thresholds={})
    for threshold in quality_thresholds:
        mask = (quality >= threshold) & enough
        grid = mask.reshape(rows, columns)
        components, count = ndlabel(grid, structure=np.ones((3, 3)))
        sizes = [int((components == index).sum()) for index in range(1, count + 1)]
        centres = []
        for index in range(1, count + 1):
            ys, xs = np.nonzero(components == index)
            centres.append([float((xs.mean() + 0.5) * block), float((ys.mean() + 0.5) * block)])
        hertz = [float(np.median(peak_hz.reshape(rows, columns)[components == index]))
                 for index in range(1, count + 1)]
        result['thresholds'][str(threshold)] = dict(
            blocks=int(mask.sum()), regions=count,
            region_sizes=sorted(sizes, reverse=True)[:8],
            region_centres=centres[:8], region_median_hz=hertz[:8])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--block', type=int, default=BLOCK, choices=[8, 16, 32])
    parser.add_argument('--segments', default=(
        '未出现0-18s:9:2,清晰22-30s:26:2,清晰34-40s:37:2,'
        '空档42-50s:46:2,侧视109-114s:111.5:2,离场114.5-116s:115.2:1,'
        '再现116.6-120s:118.3:2'))
    parser.add_argument('--thresholds', default='6,10,20,40')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    thresholds = [float(item) for item in args.thresholds.split(',') if item.strip()]
    segments = []
    for item in args.segments.split(','):
        name, start, duration = item.split(':')
        segments.append((name, float(start), float(duration)))

    reader = open_raw(args.raw)
    height, width = reader.get_size()
    report = []
    for name, start, duration in sorted(segments, key=lambda item: item[1]):
        # One analysis window in the middle of the segment, end-aligned to the
        # window length; the reader only moves forward.
        centre = start + duration / 2
        begin = int(round(centre * 1e6)) - WINDOW_US // 2
        reader.seek_time(begin)
        events = reader.load_delta_t(WINDOW_US)
        events = events[(events['t'] >= begin) & (events['t'] < begin + WINDOW_US)]
        result = analyse(events, width, height, thresholds, args.block)
        if result is None:
            continue
        result.update(segment=name, begin_s=begin / 1e6, events=int(len(events)))
        report.append(result)
        summary = {key: dict(blocks=value['blocks'], regions=value['regions'])
                   for key, value in result['thresholds'].items()}
        print(json.dumps(dict(segment=name, events=len(events), summary=summary), ensure_ascii=False))
    del reader
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(dict(written=args.out)))


if __name__ == '__main__':
    main()
