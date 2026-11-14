"""Current native contour motion replaces phase-texture shift if qualified."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from check_inverse322 import model
from run_rotor_only import SOURCE_FILES,serial
from rotor_cuda_worker import CudaRotorFrontWorker
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',default='15');p.add_argument('--batch',default='shape385_20261110');a=p.parse_args();n=a.recording
    out=ROOT.parent/f'output/{a.batch}/recording{n}';out.mkdir(parents=True,exist_ok=False)
    expected=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()]
    cache=ROOT.parent/('output/frozen324_20261109/cache' if n=='4' else 'output/frozen_tangent287_20261108/cache' if n=='3' else 'output/motion181_20261107/cache_recording15' if n=='15' else f'output/phase185_20261107/cache_recording{n}')
    manifest=json.loads((cache/'manifest.json').read_text('utf8'))['frames'] if n in ('3','4') else None
    files=sorted(set(SOURCE_FILES+['native_shape_registration.py','tools/pilot_shape385.py']));hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        q=out/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/f).read_bytes())
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,one_global_rule=True,no_parameter_search=True,NN_used=False,same_EventCD_GPU_features_both_arms=True,RAW_FPS_measured=False),indent=2),'utf8')
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    models={arm:model(worker,inverse_normal_response=True,anisotropic_response=True) for arm in ('control','candidate')};models['candidate'].front.native_shape_motion=True
    parity=0.;prefix=0.;qualified=0;outputs={arm:[] for arm in models}
    try:
        for i,wanted in enumerate(expected):
            t=wanted['t'];e=np.load(cache/manifest[i]['file']) if manifest else window(cache,t);features=worker.prepare_uncached(e,t)
            for arm,m in models.items():
                worker.cached=(t,features);r,frame=m.step(e[:0],t);frame=None;assert r['counts']['events']==wanted['counts']['events']
                if arm=='control':
                    assert r['valid']==wanted['valid'] and bool(r['coefficient'])==bool(wanted['coefficient'])
                    if r['coefficient']:
                        for h,v in r['physical'].items():
                            d=float(np.max(abs(np.asarray(v)-wanted['physical'][h])));parity=max(parity,d)
                            if t<=15.600001:prefix=max(prefix,d)
                else:qualified+=bool((r['motion'].get('estimate') or {}).get('phase_texture_not_used'))
                outputs[arm].append(r)
                with (out/f'{arm}.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,default=serial,allow_nan=False)+'\n')
            if i%30==0:print(n,i,'parity',prefix,'shape motions',qualified,flush=True)
        assert prefix<1e-7,prefix
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    finally:
        for m in models.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
        worker.cached=None;worker.close();configure_native(None)
    ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
    cases={arm:score(out/f'{arm}.jsonl',ref,origins) for arm in models}
    result=dict(cases=cases,qualified_shape_motions=qualified,control_parity=parity,scored_prefix_parity=prefix,RAW_FPS_measured=False)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(n,{a:v['error'] for a,v in cases.items()},flush=True)
