"""Add a read-only snapshot ABI to the owned bridge; upstream untouched."""
from pathlib import Path
import json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
out=WORK/'output/snapshot335_20261109';out.mkdir(exist_ok=False);src=out/'build_source';src.mkdir()
own=ROOT/'research/helix_reference';upstream=WORK/'helixtrack-main'
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in upstream.glob('*.h')}
extension=r'''
extern "C" __declspec(dllexport) int ht_snapshot_v1(void* pointer,double* output){
 try{
  auto& h=*static_cast<Handle*>(pointer);auto state=h.tracker.state();auto pose=h.tracker.pose();
  const double t=h.tracker.last_time_s();
  output[0]=double(h.first);output[1]=double(h.first)+1e6*t;
  output[2]=state.phase_at(t)-h.blades*double(pose.q(1));output[3]=state.omega;output[4]=state.alpha;
  for(int k=0;k<9;++k)output[5+k]=pose.H.data()[k];
  for(int k=0;k<6;++k)output[14+k]=pose.q(k);
  output[20]=h.tracker.rotation_sign();
  for(int k=0;k<21;++k)if(!std::isfinite(output[k]))return -2;
  return 0;
 }catch(...){return -1;}
}
'''
(src/'helix_bridge.cpp').write_text((own/'helix_bridge.cpp').read_text('utf8')+extension,'utf8')
(src/'CMakeLists.txt').write_bytes((own/'CMakeLists.txt').read_bytes())
cmake='C:/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
commands=[[cmake,'-S',str(src),'-B',str(out/'build'),'-G','Visual Studio 18 2026','-A','x64','-DHELIX_SOURCE='+str(upstream),'-DCMAKE_PREFIX_PATH=D:/Prophesee/third_party','-DOpenCV_DIR=D:/Prophesee/third_party/share/opencv4'],[cmake,'--build',str(out/'build'),'--config','Release','--parallel','2']]
for i,command in enumerate(commands):
 with (out/f'build{i}.log').open('w',encoding='utf8') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 if r.returncode:raise RuntimeError(f'Inspect build{i}.log; no dependency installation attempted')
assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in upstream.glob('*.h')}
(out/'protocol.json').write_text(json.dumps(dict(commands=commands,upstream_headers_unchanged=True,upstream_sha256=hashes,old_bridge_unchanged=True,CPU_only=True,new_dependency_installed=False,snapshot_has_no_measurement_side_effect=True),indent=2),'utf8')
print(out/'build/Release/helix_reference.dll')
