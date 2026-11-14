"""Current conic qualification vs side-line fallback; original GPU events."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from check_inverse322 import model
from run_rotor_only import SOURCE_FILES,serial
from rotor_cuda_worker import CudaRotorFrontWorker
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window
from physical_publication import evaluate
from axis_amplitude_bridge import current_axis_qualified
from score_rotor164 import score
from extend_inverse321 import sources,references

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',default='4');p.add_argument('--batch',default='switch332_20261109');a=p.parse_args();n=a.recording
    out=ROOT.parent/f'output/{a.batch}/recording{n}';out.mkdir(parents=True,exist_ok=False)
    cache=ROOT.parent/('output/frozen324_20261109/cache' if n=='4' else 'output/motion181_20261107/cache_recording15' if n=='15' else f'output/phase185_20261107/cache_recording{n}')
    source=ROOT.parent/'output/frozen324_20261109/control.jsonl' if n=='4' else sources(n)
    rows=[json.loads(s) for s in source.read_text('utf8').splitlines()]
    manifest=json.loads((cache/'manifest.json').read_text('utf8')) if n=='4' else None
    files=sorted(set(SOURCE_FILES+['tools/replay_switch327.py']));hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files if (ROOT/f).is_file()}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,original_source=str(source),NN_used=False,GPU_used=True,RAW_FPS_measured=False,
        no_recording_specific_rules=True,no_parameter_search=True,existing_arc_quality_limits_reused=True,
        original_default_not_replaced=True,inference_before_reference_reads=True),indent=2),'utf8')
    for f in hashes:
        q=out/'source_snapshot'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/f).read_bytes())
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    models=dict(control=model(worker),switch=model(worker),switch_inverse=model(worker,inverse_normal_response=True,tangential_memory=True))
    for arm,m in models.items():m.front.geometry.qualified_plane_switch=arm!='control'
    outputs={a:[] for a in models};parity=0.
    try:
        for i,old in enumerate(rows):
            t=old['t'];e=np.load(cache/manifest['frames'][i]['file']) if manifest is not None else window(cache,t);features=worker.prepare_uncached(e,t)
            for arm,m in models.items():
                worker.cached=(t,features)
                r,frame=m.step(e[:0],t);frame=None;outputs[arm].append(r)
                assert r['counts']['events']==old['counts']['events'],(arm,i,'event input mismatch')
                with (out/f'{arm}_partial.jsonl').open('a',encoding='utf8') as fh:fh.write(json.dumps(r,default=serial,allow_nan=False)+'\n')
                if arm=='control' and r['coefficient']:
                    for h,p in r['physical'].items():parity=max(parity,float(np.max(abs(np.asarray(p)-old['physical'][h]))))
            if i%45==0:print(n,i,'control parity',parity,'axis',[(arm,current_axis_qualified(rr[-1]['structure'])) for arm,rr in outputs.items()],flush=True)
        assert parity<1e-7,parity
        assert all(any(r['coefficient'] for r in rr) for rr in outputs.values()),'Empty candidate state stream'
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    finally:
        for m in models.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
        worker.cached=None;worker.close();configure_native(None)
    for arm,rr in outputs.items():(out/f'{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rr),'utf8')
    ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);scores={arm:score(out/f'{arm}.jsonl',ref,np.round(np.arange(10.8,15.6001,.2),1)) for arm in outputs}
    result=dict(cases=scores,control_parity=parity,current_axis_qualified={arm:sum(current_axis_qualified(r['structure']) for r in rr) for arm,rr in outputs.items()},
        side_does_not_require_four_recovered_centres=True,NN_used=False,RAW_FPS_measured=False)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(n,{arm:s['error'] for arm,s in scores.items()},result['current_axis_qualified'],flush=True)
