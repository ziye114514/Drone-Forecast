"""Read-only SHA256 fingerprints for the four tested input files."""
from pathlib import Path
import hashlib,json,time

root=Path(__file__).resolve().parents[2]/'output/raw_only_project'
records=[]
for sid in (6,2,3,4):
    m=json.loads((root/'runs'/f'train{sid}'/'run_manifest.json').read_text())
    path=Path(m['input_raw']);h=hashlib.sha256();start=time.perf_counter()
    with path.open('rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    r=dict(sequence=f'train{sid}',raw=str(path),bytes=path.stat().st_size,sha256=h.hexdigest(),hash_seconds=time.perf_counter()-start)
    records.append(r);print(json.dumps(r),flush=True)
(root/'raw_fingerprints.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
