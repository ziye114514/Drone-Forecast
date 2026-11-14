"""Actual native GPU public factory, default vs opt-in differential torque."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,time,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from run_rotor_only import RotorPredictor,SOURCE_FILES,serial,make_parser
from rotor_cuda_worker import CudaRotorFrontWorker
from native_history_buffer import NativeHistoryBuffer
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window
from replay_joint236 import source
from physical_publication import evaluate
from score_rotor164 import score


if __name__=='__main__':
    OUT=ROOT.parent/'output/factory300_20261109';OUT.mkdir(exist_ok=False)
    files=sorted(set(SOURCE_FILES+['tools/check_torque_factory300.py']));hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files if (ROOT/f).is_file()}
    for f in hashes:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    protocol=dict(source_sha256=hashes,NN_used=False,GPU_used=True,RAW_FPS_measured=False,
        no_independent_period_DLL_or_host_event_copy_added=True,extra_geometry_search_added=False,
        qualification_thresholds_changed=False,old_default_not_replaced=True,position_state_frozen=True,
        no_parameter_search=True,no_recording_specific_rules=True,labels_read_only_after_all_inference=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    try:RotorPredictor(torque_innovation=True)
    except ValueError:guard=True
    else:raise AssertionError('Unsupported observer accepted')
    assert any('--rotor-torque-innovation' in a.option_strings for a in make_parser()._actions)
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    all_results={};timings={};parities={};modified={};counts={}
    try:
        for n in ('13','15','16'):
            old=[json.loads(s) for s in source(n).read_text('utf8').splitlines()]
            cache=ROOT.parent/f'output/{"motion181_20261107" if n=="15" else "phase185_20261107"}/cache_recording{n}'
            models={};outputs={a:[] for a in ('control','torque')};cost={a:[] for a in outputs};delta=0.;state_delta=0.
            for arm in outputs:
                m=RotorPredictor(no_phase=True,fast_geometry=True,local_phase=True,subpixel_motion=True,entry_init=True,
                    relative_observation=True,relock_support_guard=True,border_motion=True,gap_prediction='short',reference_mode='fixed',
                    unknown_force_model='plane-adaptive',projected_fov_guard=True,torque_innovation=arm=='torque')
                m.front.current_region_search=True;m.period_worker=worker;m.front.period_estimator=worker;m.front.history_buffer=NativeHistoryBuffer();models[arm]=m
            for i,r in enumerate(old):
                t=r['t'];e=window(cache,t);features=worker.prepare_uncached(e,t)
                for arm,m in models.items():
                    worker.cached=(t,features);tick=time.perf_counter();new,frame=m.step(e[:0],t);cost[arm].append(1000*(time.perf_counter()-tick));frame=None;outputs[arm].append(new)
                    if new['coefficient']:
                        if arm=='control':
                            for h,value in new['physical'].items():delta=max(delta,float(np.max(abs(np.asarray(value)-r['physical'][h]))))
                        for k in ('position','velocity','acceleration'):state_delta=max(state_delta,float(np.max(abs(np.asarray(new['coefficient'][k])-r['coefficient'][k]))))
                if i%45==0:print(n,i,'baseline parity',delta,'current-state parity',state_delta,flush=True)
            assert max(delta,state_delta)<1e-7,(n,delta,state_delta)
            parities[n]=dict(default_forecast_max_px=delta,both_arms_current_state_max_difference=state_delta)
            for arm,rr in outputs.items():
                (OUT/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rr),'utf8')
            infos=[r['rotor_torque_innovation'] for r in outputs['torque']]
            modified[n]=sum(i.get('normal_components_modified',0)>0 for i in infos)
            counts[n]=dict(current_individual_period_frames=sum(i['current_source_count']>0 for i in infos),
                current_individual_counts={str(k):sum(i['current_source_count']==k for i in infos) for k in range(5)},
                informative_drive_frames=sum(abs(i.get('drive_change',0))>1e-12 for i in infos))
            timings[n]={arm:dict(mean_ms=float(np.mean(v)),median_ms=float(np.median(v))) for arm,v in cost.items()}
            for m in models.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
            features=None;worker.cached=None
    finally:worker.close();configure_native(None)
    # References opened only after all three complete inference runs are frozen.
    for n in ('13','15','16'):
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        all_results[n]={arm:score(OUT/f'{n}_{arm}.jsonl',ref,origins) for arm in ('control','torque')}
        print(n,{a:s['error'] for a,s in all_results[n].items()},flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(cases=all_results,parity=parities,modified_frames=modified,
        source_counts=counts,CPU_model_cost=timings,incompatible_configuration_guard=guard,protocol=protocol),indent=2),'utf8')
