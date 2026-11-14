"""Matched motion20/native geometry time support; unchanged period entry and force model."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,time,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_rotor_only import RotorPredictor,SOURCE_FILES,serial
from rotor_cuda_worker import CudaRotorFrontWorker
from native_history_buffer import NativeHistoryBuffer
from pure_visibility170 import window
from timing_recovery_front import recover
BASE=ROOT.parent/'output/polarity196_20261107'
FILES=list(dict.fromkeys(SOURCE_FILES+['regional_timing_period.py','timing_recovery_front.py','polarity_motion_quality.py','tools/live_polarity196.py']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',choices=['15','14','16','17','19','side','train2'],required=True);args=p.parse_args()
    OUT=BASE/f'recording{args.recording}';OUT.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(2)
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in FILES}
    for n in FILES:
        dst=OUT/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/n).read_bytes())
    (OUT/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,NN_used=False,labels_during_inference=False,
        thresholds_fitted_to_recordings=False,new_metric_RPM_or_thrust_claimed=False,RAW_FPS_measured=False,
        changed='Same joint displacement; ON/OFF channel disagreement increases uncertainty; same period and normal force'),indent=2),'utf-8')
    def guard(event,items):
        if event=='open' and items and isinstance(items[0],(str,bytes,os.PathLike)):
            pp=Path(os.fsdecode(items[0]))
            if pp.name in ('labels.json','references.json') or 'evaluation_data' in pp.parts:raise RuntimeError('No labels during inference')
    sys.addaudithook(guard)
    cache=ROOT.parent/f'output/{"motion181_20261107" if args.recording in ("15","19") else "phase185_20261107"}/cache_recording{args.recording}'
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True)
    models={};rows={};prepared=None;frame=None;traces=[]
    for arm in ('plane191','polarity'):
        m=RotorPredictor(no_phase=True,fast_geometry=True,local_phase=True,subpixel_motion=True,entry_init=True,
            relative_observation=True,relock_support_guard=True,border_motion=True,gap_prediction='short',reference_mode='fixed',unknown_force_model='plane')
        m.front.polarity_motion_quality=arm=='polarity';m.period_worker=worker;m.front.period_estimator=worker;m.front.history_buffer=NativeHistoryBuffer();models[arm]=m;rows[arm]=[]
    try:
        manifest=json.loads((cache/'manifest.json').read_text('utf-8'))
        for i,item in enumerate(manifest['frames']):
            t=item['t'];events=window(cache,t);prepared=worker.prepare_uncached(events,t)
            worker.cached=(t,prepared);r,frame=models['plane191'].step(events[:0],t);frame=None;rows['plane191'].append(r)
            started=time.perf_counter();adjusted=prepared;info=dict(used=True,reason='current_ONOFF_disagreement_covariance');info['recovery_ms']=1000*(time.perf_counter()-started)
            worker.cached=(t,adjusted);r,frame=models['polarity'].step(events[:0],t);frame=None;r['polarity_profile']=info;rows['polarity'].append(r)
            traces.append(dict(t=t,**info))
            if i%45==0:print(args.recording,i,info['used'],r['valid'],flush=True)
    finally:
        for m in models.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
        prepared=None;worker.cached=None;worker.close()
    summary={}
    for arm,rr in rows.items():
        (OUT/f'{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rr),'utf-8')
        summary[arm]=dict(frames=len(rr),valid=sum(r['valid'] for r in rr),regions=sum(r['body'].get('detected',False) for r in rr),position_observations=sum(r['motion'].get('current_observation',False) for r in rr))
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    (OUT/'completion.json').write_text(json.dumps(dict(source_changed=False,summary=summary,recovered_packets=sum(r['used'] for r in traces),recovery_mean_ms=float(np.mean([r['recovery_ms'] for r in traces])),RAW_FPS_measured=False),indent=2),'utf-8');print(json.dumps(summary),flush=True)
