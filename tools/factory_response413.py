"""Actual native GPU factory, one global response model and unchanged inputs."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,numpy as np,cv2
from functools import partial
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from check_inverse322 import model
from learned_rotor_predictor import LearnedRotorPredictor
from run_rotor_only import SOURCE_FILES,serial
from rotor_cuda_worker import CudaRotorFrontWorker
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window
from score_rotor164 import score
from extend_inverse321 import references

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',default='factory413_20261110');parser.add_argument('--recording',default='13');parser.add_argument('--response-model',type=Path,default=ROOT/'models/native_response_plane_v1.json');parser.add_argument('--expected',type=Path);args=parser.parse_args();n=args.recording
    out=ROOT.parent/'output'/args.batch;out.mkdir(exist_ok=False);path=args.response_model
    expected_path=args.expected or ROOT.parent/f'output/combined410_20261110/{n}_joint.jsonl'
    expected=[json.loads(s) for s in expected_path.read_text('utf8').splitlines()];baseline=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()]
    files=sorted(set(SOURCE_FILES+['tools/factory_response413.py','tools/check_inverse322.py']));hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    m=model(worker,predictor_type=partial(LearnedRotorPredictor,response_model=path),inverse_normal_response=True,anisotropic_response=True);outputs=[];parity=0.;state_delta=0.
    try:
        for i,old in enumerate(expected):
            t=old['t'];e=window(ROOT.parent/('output/motion181_20261107/cache_recording15' if n=='15' else f'output/phase185_20261107/cache_recording{n}'),t);worker.cached=(t,worker.prepare_uncached(e,t));r,frame=m.step(e[:0],t);frame=None
            assert r['valid']==old['valid'] and r['counts']['events']==old['counts']['events'] and r['body']==old['body']
            if r['coefficient']:
                for h,p in r['physical'].items():parity=max(parity,float(np.max(abs(np.asarray(p)-old['physical'][h]))))
                for k in ('position','velocity','acceleration'):state_delta=max(state_delta,float(np.max(abs(np.asarray(r['coefficient'][k])-baseline[i]['coefficient'][k]))))
            outputs.append(r)
            if i%45==0:print(i,'forecast parity',parity,'base-state delta',state_delta,flush=True)
        assert max(parity,state_delta)<1e-7,(parity,state_delta)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    finally:m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None;worker.cached=None;worker.close();configure_native(None)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in outputs),'utf8')
    result=dict(frames=len(outputs),recording=n,response_model=str(path),same_global_model_all6=True,all6_sources_excluded_from_fit=True,actual_factory_parity_px=parity,baseline_current_state_delta_px=state_delta,
        score=score(out/'predictions.jsonl',references(n),np.round(np.arange(10.8,15.6001,.2),1)),source_sha256=hashes,
        optional_author_joint_phase_required=False,RAW_FPS_certificate=False)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(result['score']['error'],flush=True)
