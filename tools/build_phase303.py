from pathlib import Path
import json,subprocess,hashlib,argparse
p=argparse.ArgumentParser();p.add_argument('--batch',default='continuous305_20261109');a=p.parse_args()
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output'/a.batch;OUT.mkdir(exist_ok=False)
src=OUT/'build_source';src.mkdir()
(src/'continuous_phase303.cpp').write_bytes((ROOT/'native/continuous_phase303.cpp').read_bytes())
(src/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.20)\nproject(continuous_phase LANGUAGES CXX)\nset(CMAKE_CXX_STANDARD 17)\nadd_library(continuous_phase SHARED continuous_phase303.cpp)\n','utf8')
cmake='C:/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
commands=[[cmake,'-S',str(src),'-B',str(OUT/'build'),'-G','Visual Studio 18 2026','-A','x64'],[cmake,'--build',str(OUT/'build'),'--config','Release','--parallel','2']]
for i,cmd in enumerate(commands):
    with (OUT/f'build{i}.log').open('w',encoding='utf8') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if r.returncode:raise RuntimeError(f'Inspect build{i}.log; no dependency installation attempted')
(OUT/'build_protocol.json').write_text(json.dumps(dict(commands=commands,upstream_source_modified=False,new_dependencies_installed=False,
    own_source_sha256=hashlib.sha256((ROOT/'native/continuous_phase303.cpp').read_bytes()).hexdigest()),indent=2),'utf8')
print(OUT/'build/Release/continuous_phase.dll')
