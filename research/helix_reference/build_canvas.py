"""Separate local DLL variant; do not replace the evaluated v2 author bridge."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent
OUT=WORK/'output/helix_author_probe_20261025'/('adaptive_canvas' if len(sys.argv)<2 else sys.argv[1]);OUT.mkdir(exist_ok=False)
source=WORK/'helixtrack-main';own=Path(__file__).parent
cmake=Path('C:/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe')
commands=[[str(cmake),'-S',str(own),'-B',str(OUT/'build'),'-G','Visual Studio 18 2026','-A','x64',
           '-DHELIX_SOURCE='+str(source),'-DCMAKE_PREFIX_PATH=D:/Prophesee/third_party',
           '-DOpenCV_DIR=D:/Prophesee/third_party/share/opencv4'],
          [str(cmake),'--build',str(OUT/'build'),'--config','Release','--parallel','2']]
fingerprints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('helixtrack*.h')}
shutil.copy2(own/'helix_bridge.cpp',OUT/'helix_bridge.cpp')
for i,cmd in enumerate(commands):
    with (OUT/f'build_{i}.log').open('x',encoding='utf-8') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
assert fingerprints=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('helixtrack*.h')}
(OUT/'provenance.json').write_text(json.dumps(dict(original_headers_unchanged=True,source_sha256=fingerprints,
    purpose='Canvas size and initial pose exposed by own ABI; author core unchanged. LOCAL ONLY.',
    author_DLL_sha256=hashlib.sha256((OUT/'build/Release/helix_reference.dll').read_bytes()).hexdigest()),indent=2),encoding='utf-8')
print('Built separate local canvas DLL',flush=True)
