"""Final integrated native-cache run, fixed172 geometry and explicit gap arms."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from run_rotor_only import RotorPredictor,SOURCE_FILES,serial
from rotor_cuda_worker import CudaRotorFrontWorker
from native_history_buffer import NativeHistoryBuffer
from pure_visibility170 import window
OUT=ROOT.parent/'output/gap173_20261107/integrated'

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--rim-fix',action='store_true');parser.add_argument('--out-name',default='integrated');args=parser.parse_args()
    if Path(args.out_name).name!=args.out_name:raise ValueError('One new output leaf name required')
    OUT=ROOT.parent/('output/rim174_20261107' if args.rim_fix else 'output/gap173_20261107')/args.out_name
    OUT.mkdir(exist_ok=False);cv2.setNumThreads(2)
    files=list(dict.fromkeys(SOURCE_FILES+['tools/live_gap173.py','tools/pure_visibility170.py']))
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    for n in files:
        p=OUT/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    (OUT/'protocol.json').write_text(json.dumps(dict(NN_used=False,labels_during_inference=False,RAW_FPS_measured=False,
        source_time_horizon_s=.4,geometry_and_native_match_recipe='172 final',
        arms=(dict(control='173 matched0.1s gap, old two-conic rim skip',short='same gap plus current parent footprint coverage before skipping') if args.rim_fix else
              dict(control='legacy random-walk missing propagation',short='matched0.1s conditional prior only for missing positions')),
        side_queries=[110.4+i*.2 for i in range(19)],train2_queries=[43.2+i*.2 for i in range(7)],source_sha256=hashes),indent=2),'utf-8')
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0]))
            if p.name=='labels.json' or 'evaluation_data' in p.parts:raise RuntimeError('No reference labels during live inference')
    sys.addaudithook(guard)
    summary={}
    for case,path,start,stop,regional in [
        ('side',ROOT.parent/'output/side_temporal164_20261101/cache_side_warm',110.,114.4,False),
        ('train2',ROOT.parent/'output/native_plane165_20261104/cache_train2_inside',42.5,44.7,True)]:
        worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,regional_seeds=regional,remote_unpack=True)
        manifest=json.loads((path/'manifest.json').read_text('utf-8'));models={};rows={};frame=None;prepared=None
        for arm,mode in [('control','short' if args.rim_fix else 'legacy'),('short','short')]:
            m=RotorPredictor(no_phase=True,fast_geometry=True,local_phase=True,subpixel_motion=True,entry_init=True,
                relative_observation=True,relock_support_guard=True,border_motion=True,legacy_visibility=False,gap_prediction=mode,reference_mode='legacy')
            m.front.geometry.reconsider_small_cores=bool(args.rim_fix and arm=='short')
            m.period_worker=worker;m.front.period_estimator=worker;m.front.history_buffer=NativeHistoryBuffer();models[arm]=m;rows[arm]=[]
        try:
            for i,item in enumerate(r for r in manifest['frames'] if start<r['t']<=stop):
                t=item['t'];e=window(path,t);prepared=worker.prepare_uncached(e,t)
                for arm,m in models.items():
                    worker.cached=(t,prepared);row,frame=m.step(e[:0],t);rows[arm].append(row);frame=None
                if i%30==0:print(case,i,flush=True)
        finally:
            for m in models.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
            prepared=None;worker.cached=None;worker.close()
        summary[case]={}
        for arm,rs in rows.items():
            (OUT/f'{case}_{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf-8')
            summary[case][arm]=dict(frames=len(rs),detected=sum(r['body'].get('detected',False) for r in rs),valid=sum(r['valid'] for r in rs))
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    (OUT/'completion.json').write_text(json.dumps(dict(summary=summary,source_changed=False),indent=2),'utf-8');print('COMPLETE',flush=True)
