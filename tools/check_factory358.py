"""Actual public predictor vs canonical same-event force candidate."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from check_inverse322 import model
from run_rotor_only import SOURCE_FILES,serial,RotorPredictor
from rotor_cuda_worker import CudaRotorFrontWorker
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window
from physical_publication import evaluate


if __name__=='__main__':
    out=ROOT.parent/'output/factory358_20261109';out.mkdir(exist_ok=False)
    files=sorted(set(SOURCE_FILES+['tools/check_factory358.py']));hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,per_arm_one_shot_cache_rearmed=True,same_original_EventCD_for_both_arms=True,RAW_decode_or_FPS_measured=False,NN_used=False),indent=2),'utf8')
    for kwargs in (dict(anisotropic_response=True),dict(visible_force_rate=True),dict(periodic_continuation=True)):
        try:RotorPredictor(**kwargs)
        except ValueError:pass
        else:raise AssertionError(('invalid configuration accepted',kwargs))
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    models=dict(control=model(worker,inverse_normal_response=True),candidate=model(worker,inverse_normal_response=True,anisotropic_response=True))
    expected={arm:[json.loads(s) for s in (ROOT.parent/('output/inverse326_20261109/13_inverse320.jsonl' if arm=='control' else 'output/force357_20261109/13.jsonl')).read_text('utf8').splitlines()] for arm in models}
    differences={arm:0. for arm in models};state_delta=0.
    try:
        for i,old in enumerate(expected['control']):
            t=old['t'];e=window(ROOT.parent/'output/phase185_20261107/cache_recording13',t);features=worker.prepare_uncached(e,t)
            for arm,m in models.items():
                worker.cached=(t,features);r,frame=m.step(e[:0],t);frame=None;wanted=expected[arm][i]
                assert r['counts']['events']==old['counts']['events'];assert r['body']['detected']==old['body']['detected'];assert r['valid']==old['valid'];assert bool(r['coefficient'])==bool(old['coefficient'])
                if r['coefficient']:
                    for h,p in evaluate(wanted,[.1,.2,.4]).items():differences[arm]=max(differences[arm],float(np.max(abs(np.array(r['physical'][h])-p))))
                    for k in ('position','velocity','acceleration'):state_delta=max(state_delta,float(np.max(abs(np.array(r['coefficient'][k])-old['coefficient'][k]))))
                with (out/f'{arm}.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,default=serial,allow_nan=False)+'\n')
            if i%30==0:print(i,differences,'state',state_delta,flush=True)
        assert max(state_delta,*differences.values())<1e-7,(differences,state_delta)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    finally:
        for m in models.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
        worker.cached=None;worker.close();configure_native(None)
    result=dict(frames=len(expected['control']),forecast_max_delta=differences,current_state_max_delta=state_delta,
        unchanged_current_input_and_detection=True,public_factory_candidate_verified=True,RAW_FPS_measured=False)
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(result,flush=True)
