"""One diagnosis: short-clip reset vs continuous RAW history, same algorithm.

Two fronts share each current GPU feature packet. A third force observer
resets at10s but uses the warm front's exact observations. This separates
physics initialization from detector/template history. No parameters change.
Not a speed benchmark and not an algorithm improvement claim.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,argparse,hashlib,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_rotor_only import RotorPredictor,serial
from rotor_cuda_worker import CudaRotorFrontWorker
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from native_history_buffer import NativeHistoryBuffer
from adaptive_response_observer import AdaptiveResponseObserver
from physical_publication import evaluate
from live_decode import native_batches
from score_rotor164 import score


def model(worker):
    m=RotorPredictor(no_phase=True,fast_geometry=True,local_phase=True,subpixel_motion=True,entry_init=True,
        relative_observation=True,relock_support_guard=True,border_motion=True,gap_prediction='short',reference_mode='fixed',
        unknown_force_model='plane-adaptive',projected_fov_guard=True)
    m.front.current_region_search=True;m.period_worker=worker;m.front.period_estimator=worker;m.front.history_buffer=NativeHistoryBuffer();return m


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',choices=['13','15','16'],required=True);args=p.parse_args();n=args.recording
    OUT=ROOT.parent/f'output/continuity266_20261108/recording{n}';OUT.mkdir(parents=True,exist_ok=False)
    files=['tools/audit_continuity266.py','run_rotor_only.py','normal_response_observer.py','adaptive_response_observer.py']
    (OUT/'protocol.json').write_text(json.dumps(dict(recording=n,raw=f'E:/dataset/experiment/{n}/media/event.raw',
        prefix=[0.,16.],reset_start_s=10.,evaluation_origins='10.8..15.6 step0.2,25queries',
        purpose='state-history diagnosis, NOT a new predictor or more accurate model claim',
        shared_current_features=True,parameters_changed=False,NN_used=False,GPU_used=True,RAW_FPS_measured=False,
        labels_during_inference=False,heldout_experiment1_opened=False,
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}),indent=2),'utf-8')
    for f in files:
        dst=OUT/'source_snapshot'/f;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/f).read_bytes())
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    warm=model(worker);cold=model(worker);force_cold=AdaptiveResponseObserver();outputs={k:[] for k in ['warm','cold','warm_front_force_reset']};snapshot=None
    try:
        for i,(t,e) in enumerate(native_batches(Path(f'E:/dataset/experiment/{n}/media/event.raw'),0.,16.,False,33333,reader_backend='cooperative')):
            features=worker.prepare_uncached(e,t);worker.cached=(t,features);r,frame=warm.step(e[:0],t);frame=None;outputs['warm'].append(r)
            if t<10.:snapshot=copy.deepcopy(r)
            else:
                worker.cached=(t,features);s,frame=cold.step(e[:0],t);frame=None;outputs['cold'].append(s)
                q=copy.deepcopy(r);pm=q['period_selection'];u=q['force'].get('input_u');u=None if u is None else np.asarray(u)
                if q['state'].get('two_point_initialization_selected_by_partial_entry'):force_cold.two_point_init=True
                c,v,f,state=force_cold.step(t,q['body'],u,.7 if u is not None else 0.,allow_gain_learning=pm.get('learning_input_current',True),
                    input_meta=pm,motion=q['motion'],structure=q['structure'])
                q.update(coefficient=c,valid=v,force=f,state=state,normal_turn_forecast=state.get('normal_turn_forecast'));q['physical']=evaluate(q,[.1,.2,.4]) if c else {};outputs['warm_front_force_reset'].append(q)
            if i%60==0:
                print(n,i,t,'warm valid',r['valid'],'observations',r['state'].get('observations'),flush=True)
                (OUT/'progress.json').write_text(json.dumps(dict(index=i,t=t,warm_valid=r['valid'],warm_observations=r['state'].get('observations'))),'utf-8')
    finally:
        for m in [warm,cold]:m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
        worker.cached=None;worker.close();configure_native(None)
    if snapshot:(OUT/'before10_state.json').write_text(json.dumps(snapshot,default=serial,indent=2),'utf-8')
    for name,rows in outputs.items():(OUT/f'{name}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rows),'utf-8')
    folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
    ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
    scores={name:score(OUT/f'{name}.jsonl',ref,origins) for name in outputs}
    diagnostics={name:dict(frames=len(rs),valid=sum(r['valid'] for r in rs),
        observation_counter_resets=sum((b['state'].get('observations') or 0)<(a['state'].get('observations') or 0) for a,b in zip(rs,rs[1:]))) for name,rs in outputs.items()}
    (OUT/'results.json').write_text(json.dumps(dict(scores=scores,diagnostics=diagnostics,not_an_algorithm_change=True),indent=2),'utf-8')
    print(n,{k:dict(available=v['available'],error=v['error']) for k,v in scores.items()},flush=True)
