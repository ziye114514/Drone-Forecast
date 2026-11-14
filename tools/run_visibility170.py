"""Frozen two-arm native-cache run: legacy versus curvature/visibility fix."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,json,hashlib,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from run_rotor_only import RotorPredictor,SOURCE_FILES,serial
from rotor_only_front import RotorOnlyFront
from rotor_cuda_worker import CudaRotorFrontWorker
from native_history_buffer import NativeHistoryBuffer
from pure_visibility170 import window
OUT=ROOT.parent/'output/geometry170_20261106/frozen_run'

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out-name',default='frozen_run');parser.add_argument('--side-only',action='store_true');args=parser.parse_args()
    if Path(args.out_name).name!=args.out_name:raise ValueError('One new folder name required')
    OUT=OUT.parent/args.out_name
    OUT.mkdir(exist_ok=False);cv2.setNumThreads(2);snap=OUT/'source_snapshot';snap.mkdir()
    names=list(dict.fromkeys(SOURCE_FILES+['tools/run_visibility170.py','tools/pure_visibility170.py']))
    hashes={}
    for n in names:
        data=(ROOT/n).read_bytes();p=snap/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);hashes[n]=hashlib.sha256(data).hexdigest()
    protocol=dict(arms=['legacy','fixed'],labels_read_during_inference=False,NN_used=False,source_time_horizon_s=.4,
        input='saved native EventCD packets; same GPU front result shared across arms',RAW_FPS_measured=False,
        side_queries_fixed=[110.4+i*.2 for i in range(9)],side_queries_extension=[112.2+i*.2 for i in range(10)],
        other_queries=[43.2+i*.2 for i in range(7)],known_review_cutoffs={'A':110.02,'B':111.02,'C':112.02,'D':114.02},
        parameters_fixed_before_forecast_scoring=True,source_sha256=hashes)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf-8')
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0]))
            if p.name=='labels.json' or 'evaluation_data' in p.parts:raise RuntimeError('No labels during inference')
    sys.addaudithook(guard)
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True)
    result={};frame=None;prepared=None
    try:
        for case,path,start,stop in [
            ('side',ROOT.parent/'output/side_temporal164_20261101/cache_side_warm',110.,114.4),
            ('train2',ROOT.parent/'output/native_plane165_20261104/cache_train2_inside',42.5,44.7)]:
            if args.side_only and case!='side':continue
            manifest=json.loads((path/'manifest.json').read_text('utf-8'))
            models={};rows={};costs={}
            for arm in ['legacy','fixed']:
                m=RotorPredictor(no_phase=True,fast_geometry=True,local_phase=True,subpixel_motion=True,entry_init=True,
                    relative_observation=True,relock_support_guard=True,border_motion=True,legacy_visibility=arm=='legacy',gap_prediction='legacy',reference_mode='legacy')
                m.period_worker=worker;m.front.period_estimator=worker;m.front.history_buffer=NativeHistoryBuffer();models[arm]=m;rows[arm]=[];costs[arm]=[]
            for index,item in enumerate(r for r in manifest['frames'] if start<r['t']<=stop):
                t=item['t'];events=window(path,t);prepared=worker.prepare_uncached(events,t)
                for arm in ['legacy','fixed']:
                    worker.cached=(t,prepared);tick=time.perf_counter();row,frame=models[arm].step(events[:0],t)
                    costs[arm].append(1000*(time.perf_counter()-tick));rows[arm].append(row);frame=None
                if index%30==0:print(case,index,round(t,3),flush=True)
            result[case]={}
            for arm,rs in rows.items():
                (OUT/f'{case}_{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf-8')
                result[case][arm]=dict(frames=len(rs),detected=sum(r['body'].get('detected',False) for r in rs),valid=sum(r['valid'] for r in rs),
                    current_four_hypotheses=sum(len(r['geometry'].get('rotors',[]))==4 for r in rs),
                    four_visible=sum(r['structure'].get('view_evidence')=='four_disks_visible' for r in rs),
                    model_only_CPU_ms_median=float(np.median(costs[arm])),model_only_CPU_ms_p95=float(np.percentile(costs[arm],95)))
                models[arm].front.previous=None;models[arm].front.geometry.count_current=None;models[arm].period_worker=None
            print(case,'DONE',json.dumps(result[case]),flush=True);models=None
        reviewed=[]
        cache=ROOT.parent/'output/side_temporal164_20261101/cache_side_warm'
        for label,t in [('A',110.02),('B',111.02),('C',112.02),('D',114.02)]:
            events=window(cache,t);prepared=worker.prepare_uncached(events,t);record=dict(label=label,t=t,arms={})
            for legacy in [True,False]:
                f=RotorOnlyFront(worker,fast_geometry=True,legacy_visibility=legacy);f.provide_phase_events=False
                worker.cached=(t,prepared);frame=f.step(events[:0],t)
                record['arms']['legacy' if legacy else 'fixed']=dict(body=frame['body'],structure=frame['structure'],geometry=frame['geometry'])
                if label=='C':np.savez_compressed(OUT/f'C_{legacy}.npz',on=frame['on'],off=frame['off'],native=frame['native_mask'],geometry_native=frame['geometry_native_mask'])
                f.previous=None;f.geometry.count_current=None;frame=None;f=None
            reviewed.append(record)
        (OUT/'reviewed.json').write_text(json.dumps(reviewed,default=serial,indent=2),'utf-8')
    finally:frame=None;prepared=None;worker.cached=None;worker.close()
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    (OUT/'completion.json').write_text(json.dumps(dict(summary=result,source_changed=False,actual_RAW_FPS_measured=False,NN_used=False),indent=2),'utf-8')
    print('COMPLETE',flush=True)
