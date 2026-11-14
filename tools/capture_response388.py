"""Current shared native-only pipeline on fixed cached source intervals.

Six original evaluation sources plus every other already-cached recording
12/14/17/18/19. Extra sources supply ONLY native pseudo-teachers. No manual
reference is opened. Same continuation rule and geometry roles for all cases.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from check_inverse322 import model
from run_rotor_only import SOURCE_FILES,serial
from rotor_cuda_worker import CudaRotorFrontWorker
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window

if __name__=='__main__':
    out=ROOT.parent/'output/response388_20261110';out.mkdir(exist_ok=False)
    files=sorted(set(SOURCE_FILES+['tools/capture_response388.py']));hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        q=out/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/f).read_bytes())
    specs={};missing=[]
    for n in ('13','15','16','11','3','4','12','14','17','18','19'):
        cache=ROOT.parent/('output/frozen324_20261109/cache' if n=='4' else 'output/frozen_tangent287_20261108/cache' if n=='3' else f'output/motion181_20261107/cache_recording{n}' if n in ('15','19') else f'output/phase185_20261107/cache_recording{n}')
        if not (cache/'manifest.json').exists():missing.append(n);continue
        manifest=json.loads((cache/'manifest.json').read_text('utf8'))['frames']
        if n in ('13','15','16','11','3','4'):
            expected=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()];stamps=[r['t'] for r in expected]
        else:stamps=[r['t'] for r in manifest if 10.<r['t']<=16.000001]
        specs[n]=dict(cache=str(cache),stamps=stamps,exact_packets=n in ('3','4'),manifest=manifest)
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,cases={n:{k:v for k,v in s.items() if k!='manifest'} for n,s in specs.items()},missing_cache_sources=missing,only_cached_EventCD=True,one_global_rule=True,manual_references_read=False,NN_used=False,continuation_enabled_for_every_source=True,RAW_FPS_measured=False),indent=2),'utf8')
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True);counts={}
    try:
        for n,spec in specs.items():
            m=model(worker,inverse_normal_response=True,anisotropic_response=True,periodic_continuation=True);cache=Path(spec['cache']);counts[n]=dict(frames=0,detected=0,valid=0,current_force=0)
            for i,t in enumerate(spec['stamps']):
                e=np.load(cache/spec['manifest'][i]['file']) if spec['exact_packets'] else window(cache,t)
                worker.cached=(t,worker.prepare_uncached(e,t));r,frame=m.step(e[:0],t);frame=None
                counts[n]['frames']+=1;counts[n]['detected']+=bool(r['body'].get('detected'));counts[n]['valid']+=r['valid'];counts[n]['current_force']+=bool(r['force'].get('active'))
                with (out/f'{n}.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,default=serial,allow_nan=False)+'\n')
                if i%60==0:print(n,i,counts[n],flush=True)
            m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None;worker.cached=None;print(n,'inference frozen',flush=True)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    finally:worker.cached=None;worker.close();configure_native(None)
    (out/'completion.json').write_text(json.dumps(dict(counts=counts,all_sources_same_algorithm=True,no_annotations_read=True,RAW_FPS_measured=False),indent=2),'utf8');print(counts,flush=True)
