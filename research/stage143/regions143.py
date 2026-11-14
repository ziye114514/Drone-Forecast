"""Stage 143: local-region multi-hypothesis flicker estimation.

Stage 141 asks one question of the whole body ROI: "what is the dominant flicker
frequency". On a quadrotor that mixes four motors, their harmonics and body
motion into a single number, which is presumably why the 142 model regressed
(48.776 -> 50.891 px at 0.4 s).

This module keeps the same pixel evidence but stops collapsing it:

  pixels -> grid cells -> frequency clusters -> hypotheses -> temporally tracked
  tracks with support, continuity and harmonic flags.

Only the pixels the installed SDK already provides are used (x, y, t,
frequency); no reconstruction, no new dependency. The body ROI comes from the
frozen 141 run, the pixels are recomputed from RAW, and nothing here feeds the
predictor yet.
"""
from pathlib import Path
import itertools
import math

import numpy as np

CELL = 24                    # grid cell edge in pixels
MIN_CELL_PIXELS = 6          # a cell needs this many frequency pixels
CLUSTER_LOG_TOLERANCE = math.log(1.02) * 2.0   # ~4% in frequency
MIN_HYPOTHESIS_PIXELS = 12
TRACK_LOG_GATE = math.log(1.10)                 # 10% frequency continuity gate
TRACK_SPATIAL_GATE = 90.0                       # px between predicted and measured centre
TRACK_EXPIRY = 0.30                              # s
ALPHA = 0.30                                     # EMA weight for a track update


def latest_per_pixel(q, width):
    if not len(q):
        return q
    pixel = q['y'].astype(np.int64) * width + q['x']
    _, index = np.unique(pixel[::-1], return_index=True)
    return q[len(q) - 1 - index]


def roi_pixels(q, bbox, t, velocity):
    """Frequency pixels inside the current body ROI, aligned to the current time."""
    if bbox is None or not len(q):
        return np.empty(0, np.float64), np.empty(0, np.float64), np.empty(0, np.float64)
    box = np.asarray(bbox, float)
    keep = ((q['frequency'] >= 60.) & (q['frequency'] <= 1200.)
            & (q['x'] >= box[0]) & (q['x'] < box[2])
            & (q['y'] >= box[1]) & (q['y'] < box[3]))
    q = q[keep]
    if not len(q):
        return np.empty(0, np.float64), np.empty(0, np.float64), np.empty(0, np.float64)
    age = t - q['t'] / 1e6
    x = q['x'].astype(float) + age * float(velocity[0])
    y = q['y'].astype(float) + age * float(velocity[1])
    return x, y, q['frequency'].astype(float)


def cells(x, y, frequency, bbox):
    """Per-cell frequency modes inside the ROI."""
    if not len(x):
        return []
    box = np.asarray(bbox, float)
    column = np.floor((x - box[0]) / CELL).astype(int)
    row = np.floor((y - box[1]) / CELL).astype(int)
    key = row * 1000 + column
    result = []
    for value in np.unique(key):
        mask = key == value
        count = int(mask.sum())
        if count < MIN_CELL_PIXELS:
            continue
        hz = frequency[mask]
        bins = np.round(np.log(hz / 60.) / math.log(1.02)).astype(int)
        histogram = np.bincount(bins)
        mode_bin = int(np.argmax(histogram))
        mode_mask = bins == mode_bin
        result.append(dict(column=int(column[mask][0]), row=int(row[mask][0]),
                           pixels=count, hz=float(np.median(hz[mode_mask])),
                           mode_fraction=float(mode_mask.mean()),
                           cx=float(x[mask].mean()), cy=float(y[mask].mean())))
    return result


def hypotheses(cell_list):
    """Cluster cells with comparable frequency into hypotheses."""
    if not cell_list:
        return []
    ordered = sorted(cell_list, key=lambda item: item['hz'])
    groups = []
    for cell in ordered:
        placed = False
        for group in groups:
            if abs(math.log(cell['hz'] / group['hz'])) <= CLUSTER_LOG_TOLERANCE:
                group['cells'].append(cell)
                weights = np.array([c['pixels'] for c in group['cells']], float)
                hz = np.array([c['hz'] for c in group['cells']], float)
                group['hz'] = float(np.average(hz, weights=weights))
                placed = True
                break
        if not placed:
            groups.append(dict(hz=cell['hz'], cells=[cell]))
    result = []
    for group in groups:
        pixels = sum(cell['pixels'] for cell in group['cells'])
        if pixels < MIN_HYPOTHESIS_PIXELS:
            continue
        weights = np.array([cell['pixels'] for cell in group['cells']], float)
        result.append(dict(
            hz=float(group['hz']), pixels=int(pixels), cells=len(group['cells']),
            centre=[float(np.average([cell['cx'] for cell in group['cells']], weights=weights)),
                    float(np.average([cell['cy'] for cell in group['cells']], weights=weights))],
            mode_fraction=float(np.average([cell['mode_fraction'] for cell in group['cells']],
                                           weights=weights))))
    result.sort(key=lambda item: -item['pixels'])
    # harmonic flags: does a sibling hypothesis sit near half or double this one?
    for item in result:
        flags = []
        for other in result:
            if other is item:
                continue
            ratio = other['hz'] / item['hz']
            if abs(math.log(ratio / 2.0)) <= CLUSTER_LOG_TOLERANCE:
                flags.append('double')
            elif abs(math.log(ratio / 0.5)) <= CLUSTER_LOG_TOLERANCE:
                flags.append('half')
        item['harmonic_flags'] = sorted(set(flags))
    return result


class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 0

    def step(self, hyps, t):
        for track in self.tracks:
            track['predicted'] = [track['centre'][0] + track['velocity'][0] * (t - track['t']),
                                  track['centre'][1] + track['velocity'][1] * (t - track['t'])]
        self.tracks = [track for track in self.tracks if t - track['t'] <= TRACK_EXPIRY]
        pairs = []
        for ti, track in enumerate(self.tracks):
            for hi, hyp in enumerate(hyps):
                dl = abs(math.log(hyp['hz'] / track['hz']))
                ds = math.hypot(hyp['centre'][0] - track['predicted'][0],
                                hyp['centre'][1] - track['predicted'][1])
                if dl <= TRACK_LOG_GATE and ds <= TRACK_SPATIAL_GATE:
                    pairs.append((dl / TRACK_LOG_GATE + ds / TRACK_SPATIAL_GATE, ti, hi))
        used_tracks, used_hyps = set(), set()
        for _cost, ti, hi in sorted(pairs):
            if ti in used_tracks or hi in used_hyps:
                continue
            track, hyp = self.tracks[ti], hyps[hi]
            dt = max(t - track['t'], 1e-3)
            velocity = [(hyp['centre'][0] - track['centre'][0]) / dt,
                        (hyp['centre'][1] - track['centre'][1]) / dt]
            track['velocity'] = [.8 * track['velocity'][0] + .2 * float(np.clip(velocity[0], -3000, 3000)),
                                 .8 * track['velocity'][1] + .2 * float(np.clip(velocity[1], -3000, 3000))]
            track['hz'] = (1 - ALPHA) * track['hz'] + ALPHA * hyp['hz']
            track['pixels'] = (1 - ALPHA) * track['pixels'] + ALPHA * hyp['pixels']
            track['centre'] = hyp['centre']
            track['t'] = t
            track['seen'] += 1
            track['harmonic_flags'] = hyp['harmonic_flags']
            track['mode_fraction'] = hyp['mode_fraction']
            used_tracks.add(ti)
            used_hyps.add(hi)
        for hi, hyp in enumerate(hyps):
            if hi in used_hyps or len(self.tracks) >= 8:
                continue
            self.tracks.append(dict(id=self.next_id, hz=hyp['hz'], pixels=float(hyp['pixels']),
                                    centre=hyp['centre'], t=t, seen=1,
                                    velocity=[0., 0.], predicted=hyp['centre'],
                                    harmonic_flags=hyp['harmonic_flags'],
                                    mode_fraction=hyp['mode_fraction'], birth_t=t))
            self.next_id += 1
        for track in self.tracks:
            support = min(1.0, track['pixels'] / 32.0)
            continuity = min(1.0, track['seen'] / 8.0)
            penalty = 0.5 if track['harmonic_flags'] else 1.0
            track['confidence'] = float(support * continuity * penalty * min(1.0, track['mode_fraction'] / .5))
        return [track for track in self.tracks if abs(track['t'] - t) < 1e-9]


def summarise(live, structure_supported):
    """Tick-level fusion; deliberately conservative and observation-only."""
    if not live:
        return dict(tracks=0, fused_hz=None, spread_ratio=None, mean_confidence=0.0,
                    dominant_confidence=0.0, harmonic_ambiguous=False)
    ordered = sorted(live, key=lambda track: -track['pixels'])
    top = ordered[:4]
    hertz = [track['hz'] for track in top]
    spread = (max(hertz) / max(min(hertz), 1e-9)) if len(hertz) > 1 else None
    return dict(tracks=len(ordered),
                fused_hz=float(np.median(hertz)),
                spread_ratio=None if spread is None else float(spread),
                mean_confidence=float(np.mean([track['confidence'] for track in top])),
                dominant_confidence=float(top[0]['confidence']),
                harmonic_ambiguous=any(track['harmonic_flags'] for track in top),
                local_tracks=[dict(id=track['id'], hz=round(track['hz'], 2),
                                   pixels=round(track['pixels'], 1), seen=track['seen'],
                                   centre=[round(track['centre'][0], 1), round(track['centre'][1], 1)],
                                   confidence=round(track['confidence'], 3),
                                   harmonic_flags=track['harmonic_flags']) for track in top])
