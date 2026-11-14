"""Actual GPU factory integration of the explicit learned-response wrapper."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--frame',choices=['disk','camera'],default='disk');parser.add_argument('--batch',default='factory392_20261110');args=parser.parse_args()
    out=ROOT.parent/'output'/args.batch;out.mkdir(exist_ok=False)
    fold=json.loads((ROOT.parent/'output/unknown382_20261110/models.json').read_text('utf8'))['LOO']['15'][args.frame];path=out/'source_excluded_model15.json';path.write_text(json.dumps(fold,indent=2),'utf8')
    expected=[json.loads(s) for s in (ROOT.parent/f'output/unknown382_20261110/15_{args.frame}.jsonl').read_text('utf8').splitlines()]
    control=[json.loads(s) for s in (ROOT.parent/'output/force357_20261109/15.jsonl').read_text('utf8').splitlines()]
    files=sorted(set(SOURCE_FILES+['tools/factory_response392.py','tools/check_inverse322.py']));hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    m=model(worker,predictor_type=partial(LearnedRotorPredictor,response_model=path),inverse_normal_response=True,anisotropic_response=True)
    parity=0.;state_delta=0.;outputs=[]
    try:
        for i,old in enumerate(expected):
            t=old['t'];e=window(ROOT.parent/'output/motion181_20261107/cache_recording15',t);worker.cached=(t,worker.prepare_uncached(e,t));r,frame=m.step(e[:0],t);frame=None
            assert r['valid']==old['valid'] and r['body']==old['body'] and r['counts']['events']==old['counts']['events']
            if r['coefficient']:
                for h,v in r['physical'].items():parity=max(parity,float(np.max(abs(np.asarray(v)-old['physical'][h]))))
                for k in ('position','velocity','acceleration'):state_delta=max(state_delta,float(np.max(abs(np.asarray(r['coefficient'][k])-control[i]['coefficient'][k]))))
            outputs.append(r)
            if i%45==0:print(i,'forecast parity',parity,'current-state change',state_delta,flush=True)
        assert max(parity,state_delta)<1e-7,(parity,state_delta)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    finally:m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None;worker.cached=None;worker.close();configure_native(None)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in outputs),'utf8')
    result=dict(frames=len(outputs),actual_factory_forecast_parity_px=parity,baseline_current_state_change_px=state_delta,
        score=score(out/'predictions.jsonl',references('15'),np.round(np.arange(10.8,15.6001,.2),1)),source_sha256=hashes,
        model_path_explicitly_selected_for_reproduction_not_by_runtime_video_ID=True,
        this_is_source_excluded15_model_not_shared_all_source_model=True,RAW_FPS_measured=False)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(result['score']['error'],flush=True)
