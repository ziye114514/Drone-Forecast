"""Read only the central directories of supplied FRED archives."""
from pathlib import Path
import json
import zipfile

for index in (0, 1, 2):
    path = Path(f'E:/FRED/{index}.zip')
    with zipfile.ZipFile(path) as archive:
        selected = [dict(name=i.filename, bytes=i.file_size, compressed=i.compress_size)
                    for i in archive.infolist()
                    if i.filename.endswith(('.hdf5', '.h5', '.raw', 'coordinates.txt'))]
    print(json.dumps(dict(archive=str(path), entries=selected), ensure_ascii=False), flush=True)
