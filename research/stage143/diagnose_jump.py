"""One-off diagnostic: does a single-tick centre discontinuity separate a real jump
from a smooth hard acceleration? Run on both known cases."""
import json
import pathlib
import sys

import numpy as np


def centres(path, t0, t1):
    out = []
    for line in pathlib.Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        t = row['t']
        if t0 <= t <= t1 and row['body'].get('detected'):
            out.append((t, np.array(row['body']['center'], float)))
    return out


def report(label, seq, dt=.02):
    if len(seq) < 8:
        print(label, 'too few', len(seq))
        return
    t = np.array([s[0] for s in seq])
    c = np.array([s[1] for s in seq])
    speed = np.linalg.norm(np.diff(c, axis=0) / dt, axis=1)
    flags = []
    for i in range(2, len(speed) - 2):
        index = [j for j in range(max(0, i - 2), min(len(speed), i + 3)) if j != i]
        med = float(np.median(speed[index]))
        if speed[i] > max(400.0, 1.5 * med):
            flags.append((round(float(t[i + 1]), 2), round(float(speed[i])), round(med, 1)))
    print(label, 'n=', len(speed), 'median speed %.0f' % float(np.median(speed)),
          'discontinuity flags:', flags[:6])


if __name__ == '__main__':
    report('continuous6', centres(sys.argv[1], 117.6, 118.8))
    report('train1', centres(sys.argv[2], 42.2, 42.7))
