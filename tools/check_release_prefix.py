"""One short current-entry RAW run and a frozen-prefix comparison, not a score sweep."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = dict(clear=(22., ROOT.parent / 'output/stage156_parallel_geometry_20261024/clear/predictions.jsonl'),
                 side=(110., ROOT.parent / 'output/stage156_parallel_geometry_20261024/side/predictions.jsonl'))
    reports = {}
    for name, (start, reference) in cases.items():
        cmd = [sys.executable, '-X', 'utf8', str(ROOT / 'forecast.py'), 'run',
               '--config', str(ROOT / 'configs/local.json'),
               '--raw', 'E:/dataset/train/6/media/event.raw', '--output', str(out / name),
               '--start', str(start), '--duration', '.4', '--profile', 'research',
               '--preview', '--verify-frames']
        with (out / (name + '.log')).open('x', encoding='utf-8') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        rows = [json.loads(line) for line in (out / name / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        old = {round(r['t'], 6): r for r in map(json.loads, reference.read_text(encoding='utf-8').splitlines())}
        different = []
        for row in rows:
            previous = old[round(row['t'], 6)]
            for field in ('body', 'coefficients', 'predictions', 'valid', 'preferred_forecast'):
                if row[field] != previous[field]: different.append(dict(t=row['t'], field=field))
            assert row['max_event_t'] is None or row['max_event_t'] <= round(row['t'] * 1e6)
        complete = json.loads((out / name / 'completion.json').read_text(encoding='utf-8'))
        assert complete['frames'] == 20 and complete['sources_unchanged']
        reports[name] = dict(frames=len(rows), prefix_identical=not different, differences=different,
                             completion=complete, reference=str(reference))
    (out / 'checks.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({name: dict(frames=r['frames'], prefix_identical=r['prefix_identical'],
                               differences=len(r['differences'])) for name, r in reports.items()}))
    return 0 if all(r['prefix_identical'] for r in reports.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
