"""Actual public factory enabled option vs frozen289 source predictions."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_rotor_only import RotorPredictor,SOURCE_FILES
from rotor_cuda_worker import CudaRotorFrontWorker
from native_history_buffer import NativeHistoryBuffer
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window
from physical_publication import evaluate

if __name__=='__main__':
    OUT=ROOT.parent/'output/factory292_20261108';OUT.mkdir(exist_ok=False)
    files=sorted(set(SOURCE_FILES+['tools/check_tangent_factory292.py']));hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files if (ROOT/f).is_file()}
    (OUT/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,NN_used=False,GPU_used=True,
        purpose='factory integration parity, not another parameter search or FPS test',runtime_option='rotor-tangential-memory'),indent=2),'utf8')
    try:RotorPredictor(tangential_memory=True)
    except ValueError:guard=True
    else:raise AssertionError('incompatible factory configuration accepted')
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    m=RotorPredictor(no_phase=True,fast_geometry=True,local_phase=True,subpixel_motion=True,entry_init=True,
        relative_observation=True,relock_support_guard=True,border_motion=True,gap_prediction='short',reference_mode='fixed',
        unknown_force_model='plane-adaptive',projected_fov_guard=True,tangential_memory=True)
    m.front.current_region_search=True;m.period_worker=worker;m.front.period_estimator=worker;m.front.history_buffer=NativeHistoryBuffer()
    rows=[json.loads(s) for s in (ROOT.parent/'output/identifiable289_20261108/15_tangent.jsonl').read_text('utf8').splitlines()]
    cache=ROOT.parent/'output/motion181_20261107/cache_recording15';delta=0.;state_delta=0.;rate_delta=0.
    try:
        for i,old in enumerate(rows):
            t=old['t'];e=window(cache,t);worker.cached=(t,worker.prepare_uncached(e,t));r,frame=m.step(e[:0],t);frame=None
            if r['coefficient']:
                for h in [.1,.2,.4]:
                    a=np.asarray(evaluate(r,[h])[str(h)]);b=np.asarray(evaluate(old,[h])[str(h)]);delta=max(delta,float(np.max(abs(a-b))))
                for k in ['position','velocity','acceleration']:state_delta=max(state_delta,float(np.max(abs(np.asarray(r['coefficient'][k])-old['coefficient'][k]))))
            a=r['tangential_response_memory'].get('effective_response_s_inv');b=old['tangential_response_memory'].get('effective_response_s_inv')
            if a is not None and b is not None:rate_delta=max(rate_delta,abs(a-b))
            else:assert (a is None)==(b is None)
            if i%45==0:print(i,'prediction parity',delta,'current state parity',state_delta,flush=True)
        assert max(delta,state_delta,rate_delta)<1e-7,(delta,state_delta,rate_delta)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    finally:
        m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None;worker.cached=None;worker.close();configure_native(None)
    result=dict(frames=len(rows),actual_factory_prediction_max_px=delta,current_state_max_difference=state_delta,
        learned_response_max_difference=rate_delta,incompatible_configuration_rejected_before_GPU=guard,
        old_default_unchanged=True,RAW_FPS_measured=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(result,flush=True)
