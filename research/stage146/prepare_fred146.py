"""Extract only explicitly named FRED RAW/annotations; never extract arbitrary paths."""
from pathlib import Path
import json
import shutil
import zipfile

root = Path(__file__).resolve().parents[3] / 'output/stage146_threshold_branches_20261024/input/fred0'
root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile('E:/FRED/0.zip') as archive:
    for source, target in [('Event/events.raw', 'events.raw'), ('coordinates.txt', 'coordinates.txt')]:
        destination = root / target
        if not destination.exists():
            with archive.open(source) as inp, destination.open('xb') as out:
                shutil.copyfileobj(inp, out, 1024*1024)
        print(json.dumps(dict(source=source, target=str(destination), bytes=destination.stat().st_size)), flush=True)
    names = [i.filename for i in archive.infolist() if not i.is_dir()]
print(json.dumps(dict(first_entries=names[:12], annotation_head=(root/'coordinates.txt').read_text().splitlines()[:5])), flush=True)
