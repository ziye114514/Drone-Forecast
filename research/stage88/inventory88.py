"""Read-only FRED archive/label inventory; writes only within project output."""
from pathlib import Path
import collections, datetime, hashlib, json, zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent / 'output/stage88_causal_event_controls_20261022'
OLD = ROOT.parent / 'output/stage7_localization_budget_v3_20261017'


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def parse(text):
    rows = []; bad = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip(): continue
        try:
            ts, rest = line.split(':', 1); fields = [v.strip() for v in rest.split(',')]
            box = list(map(float, fields[:4])); t = float(ts)
            assert len(box) == 4 and np.isfinite([t, *box]).all()
            identity = str(int(float(fields[4]))) if len(fields) >= 5 else None
            rows.append(dict(t=t, bbox=box, identity=identity, line=number,
                             valid=box[2]>box[0] and box[3]>box[1]))
        except Exception as exc: bad.append(dict(line=number, error=str(exc)))
    return sorted(rows, key=lambda r: r['t']), bad


def unique_rows(rows):
    """No invented association: discard every ambiguous same-time observation."""
    groups = collections.defaultdict(list)
    for r in rows: groups[r['t']].append(r)
    return [rs[0] for _, rs in sorted(groups.items()) if len(rs)==1 and rs[0]['valid']]


def main():
    assert not (OUT/'inventory88.json').exists(), 'Do not overwrite an audit'
    past = {r['sequence']: r for r in load(OLD/'local_inventory.json')}
    split_dir = OLD/'upstream/FRED/dataset_splits/canonical'
    splits = {s: {int(x.strip().strip('/')) for x in (split_dir/f'{s}_split.txt').read_text().splitlines() if x.strip()}
              for s in ('train','test')}
    summaries = []
    for seq in range(11):
        archive = Path(f'E:/FRED/{seq}.zip')
        with zipfile.ZipFile(archive) as z:
            annotations = [n for n in z.namelist() if Path(n).name=='coordinates.txt']
            assert len(annotations)==1
            content = z.read(annotations[0]); rows, bad = parse(content.decode('utf-8-sig'))
            assert not bad, (seq, bad)
            dest = OUT/'annotations'/f'{seq}.txt'; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(content)
            assert sha(dest)==next(r['sha256'] for r in past[seq]['annotations'] if r['name']=='coordinates.txt')
            raw = Path(past[seq]['raw']); info = z.getinfo(past[seq]['raw_member'])
            assert raw.stat().st_size==info.file_size and sha(raw)==past[seq]['raw_sha256']
            unique = unique_rows(rows); valid=[r for r in rows if r['valid']]
            sizes=np.array([[r['bbox'][2]-r['bbox'][0],r['bbox'][3]-r['bbox'][1]] for r in valid])
            interval = [r for r in unique if 20<=r['t']<=50.4]
            counts=collections.Counter(r['t'] for r in rows)
            summaries.append(dict(sequence=seq, role='official_test_subset_previously_inspected' if seq in splits['test'] else 'development_train',
                archive=str(archive),archive_bytes=archive.stat().st_size,raw=str(raw),raw_bytes=raw.stat().st_size,
                raw_sha256=past[seq]['raw_sha256'],raw_crc32=info.CRC,annotation_member=annotations[0],annotation_sha256=sha(dest),
                total_rows=len(rows),unique_valid_rows=len(unique),invalid_rows=sum(not r['valid'] for r in rows),
                ambiguous_timestamps=sum(v>1 for v in counts.values()),identity_available=all(r['identity'] is not None for r in rows),
                first_t=valid[0]['t'],last_t=valid[-1]['t'],median_width_height=np.median(sizes,axis=0).tolist(),
                interval_unique_rows=len(interval),interval_min_side24=sum(min(r['bbox'][2]-r['bbox'][0],r['bbox'][3]-r['bbox'][1])>=24 for r in interval)))
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sequences=summaries,
        official_split_commit='2bf89c5376eda528431b62d6c60f2c13d8f95ab4',
        split_hashes={s:sha(split_dir/f'{s}_split.txt') for s in splits},official_test_total=len(splits['test']),
        official_test_local=sorted(splits['test'] & set(past)),
        recommendation='Enough for fixed pilot, not full benchmark; more unseen official test sequences improve diversity, not RPM/pose truth.',
        download_priority_ids=sorted(splits['test']-set(past))[:8],
        no_rotor_rpm_pose_thrust_truth=True,source_archives_unchanged=True)
    save(OUT/'inventory88.json',result)
    print(json.dumps(dict(local_sequences=len(summaries),local_test=result['official_test_local'],official_test_total=len(splits['test']),
        valid_rows=sum(r['unique_valid_rows'] for r in summaries),next_test_ids=result['download_priority_ids']),ensure_ascii=False),flush=True)


if __name__=='__main__': main()
