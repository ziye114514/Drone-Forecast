"""Public RAW factory parity vs frozen321; GPU features reused exactly."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_rotor_only import RotorPredictor,SOURCE_FILES,serial
from rotor_cuda_worker import CudaRotorFrontWorker
from native_history_buffer import NativeHistoryBuffer
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window
from physical_publication import evaluate

def model(worker,predictor_type=RotorPredictor,**kwargs):
    m=predictor_type(no_phase=True,fast_geometry=True,local_phase=True,subpixel_motion=True,entry_init=True,
        relative_observation=True,relock_support_guard=True,border_motion=True,gap_prediction='short',reference_mode='fixed',
        unknown_force_model='plane-adaptive',projected_fov_guard=True,**kwargs)
    m.front.current_region_search=True;m.period_worker=worker;m.front.period_estimator=worker;m.front.history_buffer=NativeHistoryBuffer();return m

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',default='factory331_20261109');args=parser.parse_args()
    out=ROOT.parent/'output'/args.batch;out.mkdir(exist_ok=False)
    files=sorted(set(SOURCE_FILES+['tools/check_inverse322.py']));hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files if (ROOT/f).is_file()}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,NN_used=False,GPU_used=True,RAW_FPS_measured=False,
        purpose='actual factory integration, not parameter tuning',same_event_features_shared_between_control_and_candidate=True),indent=2),'utf8')
    for f in hashes:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    try:RotorPredictor(inverse_normal_response=True)
    except ValueError:guard=True
    else:raise AssertionError('incompatible factory configuration accepted')
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    models=dict(control=model(worker),inverse320=model(worker,inverse_normal_response=True),combined=model(worker,inverse_normal_response=True,tangential_memory=True))
    rows={a:[json.loads(s) for s in (ROOT.parent/f'output/inverse326_20261109/15_{a}.jsonl').read_text('utf8').splitlines()] for a in models}
    outputs={a:[] for a in models};differences={a:0. for a in models};state_delta=0.
    try:
        for i,old in enumerate(rows['control']):
            t=old['t'];e=window(ROOT.parent/'output/motion181_20261107/cache_recording15',t);features=worker.prepare_uncached(e,t)
            for a,m in models.items():
                # The worker cache is one-shot. Re-arm it for EVERY branch.
                worker.cached=(t,features)
                r,frame=m.step(e[:0],t);frame=None;outputs[a].append(r)
                assert r['counts']['events']==old['counts']['events'],(a,i,'event input mismatch')
                assert bool(r['coefficient'])==bool(old['coefficient']),(a,i,'state presence mismatch')
                assert r['valid']==old['valid'],(a,i,'validity mismatch')
                assert r['body']['detected']==old['body']['detected'],(a,i,'detection mismatch')
                if r['coefficient']:
                    for h in [.1,.2,.4]:differences[a]=max(differences[a],float(np.max(abs(np.asarray(evaluate(r,[h])[str(h)])-evaluate(rows[a][i],[h])[str(h)]))))
                    for k in ('position','velocity','acceleration'):state_delta=max(state_delta,float(np.max(abs(np.asarray(r['coefficient'][k])-old['coefficient'][k]))))
            if i%30==0:print(i,'forecast parity',differences,'state',state_delta,flush=True)
        assert max(state_delta,*differences.values())<1e-7,(differences,state_delta)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    finally:
        for m in models.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
        worker.cached=None;worker.close();configure_native(None)
    for a,rr in outputs.items():(out/f'{a}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rr),'utf8')
    result=dict(frames=len(rows['control']),actual_factory_forecast_max_delta=differences,current_state_max_delta=state_delta,
        incompatible_configuration_rejected_before_GPU=guard,old_default_unchanged=True,NN_used=False,RAW_FPS_measured=False)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(result,flush=True)
