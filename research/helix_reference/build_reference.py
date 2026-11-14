"""Local, non-distributed author-core bridge; use existing VS/SDK tools."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent
OUT=WORK/'output/helix_author_probe_20261025';OUT.mkdir(exist_ok=False)
source=WORK/'helixtrack-main'
cmake=Path('C:/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe')
commands=[[str(cmake),'-S',str(Path(__file__).parent),'-B',str(OUT/'build'),'-G','Visual Studio 18 2026','-A','x64',
           '-DHELIX_SOURCE='+str(source),'-DCMAKE_PREFIX_PATH=D:/Prophesee/third_party',
           '-DOpenCV_DIR=D:/Prophesee/third_party/share/opencv4'],
          [str(cmake),'--build',str(OUT/'build'),'--config','Release','--parallel','2']]
fingerprints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('helixtrack*.h')}
(OUT/'provenance.json').write_text(json.dumps(dict(author_checkout=str(source),source_sha256=fingerprints,
    header_modified=False,license_file_present=(source/'LICENSE').exists(),redistribution=False,
    purpose='Execute existing author tracking core via local ABI. Not full paper reproduction; no author code/compiled DLL distributed.'),indent=2),encoding='utf-8')
for i,cmd in enumerate(commands):
    with (OUT/f'build_{i}.log').open('x',encoding='utf-8') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
assert fingerprints=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('helixtrack*.h')}
print(str(OUT/'build/Release/helix_reference.dll'),flush=True)
