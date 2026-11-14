"""Small read-only FRED ZIP selection. Never extracts arbitrary archive paths."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / 'output/fred_transfer_20261029'
SEQUENCES = (0, 1, 2)


def parse(text):
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        stamp, fields = line.split(':', 1)
        values = [float(x) for x in fields.split(',')]
        if len(values) not in (4, 5, 6):
            raise ValueError((number, len(values)))
        box = values[:4]
        if not np.isfinite([float(stamp), *values]).all():
            raise ValueError(number)
        rows.append(dict(t=float(stamp), bbox=box,
                         identity=int(values[4]) if len(values) >= 5 else None))
    return rows


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('Frozen selection already exists')
    if shutil.disk_usage(OUT).free < 3 * 1024**3:
        raise RuntimeError('Less than 3 GiB free')
    selections = []
    # Fixed transfer pilot, not a new unseen official test split. Select from
    # annotation coverage/size only, before generating or scoring predictions.
    for seq in SEQUENCES:
        archive = Path(f'E:/FRED/{seq}.zip')
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            annotation = next(n for n in names if Path(n).name == 'coordinates.txt')
            raw = next(n for n in names if n.endswith('/Event/events.raw') or n == 'Event/events.raw')
            content = z.read(annotation)
            rows = parse(content.decode('utf-8-sig'))
            times = np.array([r['t'] for r in rows])
            if (np.diff(times) < 0).any():
                raise ValueError('Nonmonotonic annotation times')
            unique = [r for r in rows if sum(abs(times-r['t']) < 1e-8) == 1]
            # First 4-second interval at/after 20s with >=90 labels, no label
            # gap >0.134s and >=80% fully in-FOV, min-side >=16px boxes.
            candidates = []
            for start in range(20, 110):
                group = [r for r in unique if start <= r['t'] <= start+4.4]
                good = [r for r in group if 0 < r['bbox'][0] < r['bbox'][2] < 1280
                        and 0 < r['bbox'][1] < r['bbox'][3] < 720
                        and min(np.subtract(r['bbox'][2:], r['bbox'][:2])) >= 16]
                max_gap = max(np.diff([r['t'] for r in group]), default=99.)
                if len(group) >= 90 and len(good) >= .8*len(group) and max_gap <= .134:
                    candidates.append((start, group, good))
            if not candidates:
                raise ValueError(f'No eligible interval for sequence {seq}')
            start, group, good = candidates[0]
            dest = OUT / 'input' / str(seq)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / 'coordinates.txt').write_bytes(content)
            with z.open(raw) as src, (dest / 'events.raw').open('wb') as dst:
                shutil.copyfileobj(src, dst, 1024*1024)
            frame_names = [n for n in names if '/Event/Frames/' in '/'+n and n.endswith('.png')]
            def frame_time(n):
                return int(Path(n).stem.rsplit('_', 1)[1]) / 1e6
            samples = []
            for t in (start, start+2, start+4):
                nearest = min(frame_names, key=lambda n: abs(frame_time(n)-t))
                dst = dest / Path(nearest).name
                dst.write_bytes(z.read(nearest))
                samples.append(dict(path=str(dst), t=frame_time(nearest)))
            selections.append(dict(sequence=seq, start=start, duration=4., score_horizon=.4,
                raw=str(dest/'events.raw'), raw_bytes=(dest/'events.raw').stat().st_size,
                raw_sha256=sha(dest/'events.raw'), annotation=str(dest/'coordinates.txt'),
                annotation_sha256=sha(dest/'coordinates.txt'), annotation_fields=len(content.decode('utf-8-sig').splitlines()[0].split(':',1)[1].split(',')),
                labels_in_start_through_end_plus_horizon=len(group),
                eligible_in_fov_labels=len(good), preview_frames=samples,
                annotation_median_dt=float(np.median(np.diff(times))),
                archive=str(archive), raw_member=raw))
    protocol = dict(name='FRED local legacy transfer pilot, not complete official benchmark',
        selection='Earliest integer 4s interval >=20s meeting frozen label coverage/size gates',
        timestamp_interpretation='Decimal seconds; verify against native event-frame integer microseconds',
        labels_used_for_selection_only=True, labels_not_available_to_RAW_predictor=True,
        target_horizon_s=.4, evaluation_origin_hz=5, annotation_max_interpolation_gap_s=.134,
        sequences=selections)
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2), 'utf-8')
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == '__main__':
    main()
