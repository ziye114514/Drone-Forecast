"""Takeover backup before following the rotor-only handoff (2026-11-01).

Zips source, docs, small reports and the machine handoff with a per-file
SHA256 manifest. RAW, videos, virtual environments, the historical 10.7GB
archive and model weights are NOT copied; weights are recorded as hashes so
drift is detectable without duplicating binaries.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT/'event_drone_forecast'
STAMP='20261101_pre_rotor_only_takeover'
OUT=ROOT/'backup'/STAMP
SKIP_DIRS={'.cache','__pycache__','build','external','models','Ultralytics','.venv','.venv-event311'}
SKIP_SUFFIX={'.pyc','.pt','.onnx','.engine','.mp4','.avi','.zip','.raw','.hdf5','.npy'}
SINGLE_FILES=['WORKLOG.md','PROJECT_STATUS_20261018.md','PROJECT_STATUS_20261019.md',
    'PROJECT_STATUS_20261020.md','PROJECT_STATUS_20261021.md','PROJECT_STATUS_20261022.md']
EXTRA_FILES=['output/checkpoints/latest_handoff.json',
    'output/checkpoints/before_rotor_only_directive_20261101.json',
    'output/checkpoints/next_rotor_input_20261029.md',
    'output/结果入口.md','output/查看实验结果.html','output/README.md']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def collect():
    files=[]
    for path in PROJECT.rglob('*'):
        if not path.is_file():continue
        if any(part in SKIP_DIRS for part in path.parts):continue
        if path.suffix.lower() in SKIP_SUFFIX:continue
        files.append((path,path.relative_to(ROOT)))
    for name in SINGLE_FILES:
        path=ROOT/name
        if path.is_file():files.append((path,Path(name)))
    for name in EXTRA_FILES:
        path=ROOT/name
        if path.is_file():files.append((path,Path(name)))
    return sorted(files,key=lambda item:str(item[1]))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).isoformat()
    files=collect()
    archive=OUT/f'source_{STAMP}.zip'
    records={};total=0
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as handle:
        for path,relative in files:
            data=path.read_bytes();total+=len(data)
            handle.writestr(str(relative).replace('\\','/'),data)
            records[str(relative).replace('\\','/')]=dict(sha256=sha(data),bytes=len(data),
                mtime=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat())
    with zipfile.ZipFile(archive) as handle:
        bad=handle.testzip()
        if bad:raise ValueError(f'zip CRC failure at {bad}')
        stored=set(handle.namelist())
    missing=set(records)-stored
    if missing:raise ValueError(f'missing entries {sorted(missing)[:5]}')
    weights={}
    for path in sorted((PROJECT/'models').rglob('*')):
        if path.is_file():
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
            weights[str(path.relative_to(ROOT))]=dict(sha256=digest.hexdigest(),bytes=path.stat().st_size)
    manifest=dict(created_utc=stamp,purpose='pre-takeover restore point for ROTOR_ONLY_HANDOFF_20261101',
        archive=str(archive),archive_sha256=sha(archive.read_bytes()),
        archive_bytes=archive.stat().st_size,file_count=len(records),uncompressed_bytes=total,
        excluded=dict(directories=sorted(SKIP_DIRS),suffixes=sorted(SKIP_SUFFIX),
            note='RAW, videos, venv, the historical output archive and model weights are not copied'),
        weights_hash_only=weights,files=records)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),'utf-8')
    print(json.dumps({k:manifest[k] for k in ('archive','archive_sha256','archive_bytes',
        'file_count','uncompressed_bytes')},indent=2,ensure_ascii=False))
    print('weights hashed:',len(weights))


if __name__=='__main__':main()
