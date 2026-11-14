"""Rebuild the existing local author bridge, CPU only; no dependencies installed."""
from pathlib import Path
import hashlib,json,subprocess,os
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
OUT=WORK/'output/helix_cpu232_20261108';OUT.mkdir(exist_ok=False)
source=WORK/'helixtrack-main';own=ROOT/'research/helix_reference'
cmake=Path('C:/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe')
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.h')}
commands=[[str(cmake),'-S',str(own),'-B',str(OUT/'build'),'-G','Visual Studio 18 2026','-A','x64',
    '-DHELIX_SOURCE='+str(source),'-DCMAKE_PREFIX_PATH=D:/Prophesee/third_party','-DOpenCV_DIR=D:/Prophesee/third_party/share/opencv4'],
    [str(cmake),'--build',str(OUT/'build'),'--config','Release','--parallel','2']]
(OUT/'protocol.json').write_text(json.dumps(dict(author_source=str(source),author_headers_sha256=hashes,
    author_headers_modified=False,own_bridge_modified=True,GPU_enabled=False,third_party_redistribution=False,
    compiled_author_core_is_not_complete_paper_reproduction=True,commands=commands),indent=2),'utf-8')
for name in ['CMakeLists.txt','helix_bridge.cpp','reference_api.py']:
    dst=OUT/'source_snapshot'/name;dst.parent.mkdir(exist_ok=True);dst.write_bytes((own/name).read_bytes())
for i,cmd in enumerate(commands):
    with (OUT/f'build_{i}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:raise RuntimeError(f'Existing tools failed; inspect build_{i}.log. No installation attempted')
assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.h')}
print(OUT/'build/Release/helix_reference.dll',flush=True)
