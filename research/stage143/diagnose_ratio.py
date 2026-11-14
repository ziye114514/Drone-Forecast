"""Diagnostic: is the filtered velocity lagging the measured per-tick speed?

Replacement should fire when the filter under-reports a fast (or jump-polluted)
measurement, not when it correctly follows a hard acceleration. This prints
|v_filtered| / median(recent per-tick measured speed) for the ticks the current
rule replaced on both windows.
"""
import json
import pathlib
import sys

import numpy as np


def load(path):
    return [json.loads(line) for line in
            pathlib.Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def analyse(path, label):
    rows = load(path)
    centres = [(r['t'], np.array(r['body']['center'], float)) for r in rows if r['body'].get('detected')]
    ratios = []
    speeds = []
    for index in range(6, len(rows)):
        row = rows[index]
        coefficient = row['coefficients'].get('optimized_geometry')
        if coefficient is None:
            continue
        current = row['body']
        if not current.get('detected'):
            continue
        recent = [c for t, c in centres if t <= row['t'] and t >= row['t'] - .12]
        if len(recent) < 3:
            continue
        steps = np.diff(np.array(recent), axis=0)
        if not len(steps):
            continue
        measured = float(np.median(np.linalg.norm(steps, axis=1) / .02))
        filtered = float(np.linalg.norm(np.asarray(coefficient.get('velocity', [0., 0.]))))
        speeds.append((row['t'], filtered, measured))
    print(label, 'ticks', len(speeds))
    for t, filtered, measured in speeds:
        if measured < 50:
            continue
        ratio = filtered / measured
        if ratio < .7:
            print('  t=%.2f filtered %.0f measured %.0f ratio %.2f' % (t, filtered, measured, ratio))


if __name__ == '__main__':
    analyse(sys.argv[1], 'continuous6 (guard helped)')
    analyse(sys.argv[2], 'train1 (guard failed)')
