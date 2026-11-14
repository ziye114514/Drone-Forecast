"""Stage 16: continuous burst-based blade detector (effqpdet idea) with scoring.

The detector looks at the last 100 ms of events, bins them into B x B pixel blocks
and 1 ms time bins, and asks whether any block's event rate has a periodic burst
in the blade-pass band. That statistic does not need ON and OFF to coincide at the
same pixel, which is the gate that fails in side views.

Per tick it reports qualifying blocks, the connected regions they form, and up to
four local maxima inside those regions as rotor-disk candidates. A separate
score.py joins these ticks with the archived sparse present/absent states.

    python research/stage16/blade_detector.py --raw <RAW> --start 0 --duration 35 \
        --output output/stage16_blade_detector_20261017/run_0_35
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
from scipy.ndimage import label as ndlabel, maximum_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from event_drone_forecast.raw_io import open_raw, raw_chunks  # noqa: E402

STEP_US = 100_000
BIN_US = 1000
BINS = STEP_US // BIN_US
FMIN, FMAX = 60.0, 1600.0
MIN_EVENTS = 60
QUALITY = 20.0


def json_text(value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=convert)


def fingers():
    paths = [Path(__file__), *(ROOT / 'src/event_drone_forecast').glob('*.py')]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


class BurstDetector:
    def __init__(self, width, height, block=8):
        self.width, self.height, self.block = width, height, block
        self.rows, self.columns = height // block, width // block
        self.frequencies = np.fft.rfftfreq(BINS, d=BIN_US / 1e6)
        self.band = (self.frequencies >= FMIN) & (self.frequencies <= FMAX)
        self.band_frequencies = self.frequencies[self.band]

    def step(self, events):
        if not len(events):
            return dict(blocks_ok=0, regions=[], peaks=[])
        start = int(events['t'].min()) // BIN_US * BIN_US
        bin_index = ((events['t'].astype(np.int64) - start) // BIN_US).astype(np.int32)
        block_y = (events['y'].astype(np.int32) // self.block)
        block_x = (events['x'].astype(np.int32) // self.block)
        keep = (bin_index >= 0) & (bin_index < BINS) & (block_y < self.rows) & (block_x < self.columns)
        flat_block = block_y[keep] * self.columns + block_x[keep]
        counts = np.bincount(flat_block * BINS + bin_index[keep],
                             minlength=self.rows * self.columns * BINS).reshape(
            self.rows, self.columns, BINS).astype(np.float32)
        totals = counts.sum(axis=2)
        series = counts.reshape(-1, BINS)
        series = series - series.mean(axis=1, keepdims=True)
        spectrum = np.abs(np.fft.rfft(series * np.hanning(BINS), axis=1)) ** 2
        band_power = spectrum[:, self.band]
        peak_index = np.argmax(band_power, axis=1)
        peak_power = band_power[np.arange(band_power.shape[0]), peak_index]
        median_power = np.median(band_power, axis=1)
        quality = (peak_power / np.maximum(median_power, 1e-9)).reshape(self.rows, self.columns)
        peak_hz = self.band_frequencies[peak_index].reshape(self.rows, self.columns)
        enough = totals >= MIN_EVENTS
        mask = (quality >= QUALITY) & enough
        grid, count = ndlabel(mask, structure=np.ones((3, 3)))
        regions = []
        for index in range(1, count + 1):
            ys, xs = np.nonzero(grid == index)
            if len(xs) < 2:
                continue
            regions.append(dict(blocks=int(len(xs)),
                                centre=[float((xs.mean() + 0.5) * self.block),
                                        float((ys.mean() + 0.5) * self.block)],
                                median_hz=float(np.median(peak_hz[grid == index]))))
        smoothed = maximum_filter(quality * mask, size=3)
        peaks_mask = (quality * mask == smoothed) & mask
        ys, xs = np.nonzero(peaks_mask)
        order = np.argsort(-quality[ys, xs])[:4]
        peaks = [dict(centre=[float((xs[k] + 0.5) * self.block), float((ys[k] + 0.5) * self.block)],
                      quality=float(quality[ys[k], xs[k]]), hz=float(peak_hz[ys[k], xs[k]]))
                 for k in order]
        return dict(blocks_ok=int(mask.sum()), regions=regions, peaks=peaks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--duration', type=float, default=35.)
    parser.add_argument('--output', required=True)
    parser.add_argument('--block', type=int, default=8, choices=[8, 16, 32])
    args = parser.parse_args()

    folder = Path(args.output).resolve()
    if folder.exists():
        raise SystemExit('output directory already exists; choose a new one')
    folder.mkdir(parents=True, exist_ok=False)
    protocol = dict(stage=16, detector='burst-based blade blocks (effqpdet idea)',
                    raw=str(Path(args.raw).resolve()), start_s=args.start,
                    duration_s=args.duration, step_us=STEP_US, bin_us=BIN_US,
                    block=args.block, fmin_hz=FMIN, fmax_hz=FMAX,
                    min_events_per_block=MIN_EVENTS, quality_threshold=QUALITY,
                    causal=True, cold_start=True, source_hashes=fingers(),
                    note='One non-overlapping 100 ms window per tick; the detector never looks ahead.')
    (folder / 'protocol.json').write_text(json_text(protocol), encoding='utf-8')

    reader = open_raw(args.raw)
    height, width = reader.get_size()
    detector = BurstDetector(width, height, args.block)
    started = time.perf_counter()
    ticks = detected = 0
    detected_ms = []
    with (folder / 'ticks.jsonl').open('x', encoding='utf-8') as stream:
        for stamp, events in raw_chunks(reader, args.start, args.duration, STEP_US):
            began = time.perf_counter()
            result = detector.step(events)
            detected_ms.append((time.perf_counter() - began) * 1000.)
            stream.write(json_text(dict(t=stamp, events=int(len(events)), **result)) + '\n')
            ticks += 1
            detected += bool(result['regions'])
    del reader
    summary = dict(ticks=ticks, ticks_with_region=detected,
                   detection_rate=detected / max(ticks, 1),
                   detect_ms_median=float(np.median(detected_ms)) if detected_ms else None,
                   detect_ms_p95=float(np.percentile(detected_ms, 95)) if detected_ms else None,
                   wall_s=time.perf_counter() - started,
                   interpretation='Burst activity only; a region is not proven to be a rotor disk '
                                  'and four regions are not four motors.')
    (folder / 'summary.json').write_text(json_text(summary), encoding='utf-8')
    print(json_text(summary))


if __name__ == '__main__':
    main()
