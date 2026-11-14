"""Restore the existing small-region seed recipe for the second recording."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_rotor_only import RotorPredictor,SOURCE_FILES,serial
from rotor_cuda_worker import CudaRotorFrontWorker
from native_history_buffer import NativeHistoryBuffer
from pure_visibility170 import window
OUT=ROOT.parent/'output/geometry170_20261106/regional_run'
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out-name',default='regional_run');args=parser.parse_args()
    if Path(args.out_name).name!=args.out_name:raise ValueError('One new folder name required')
    OUT=OUT.parent/args.out_name
    OUT.mkdir(exist_ok=False);cv2.setNumThreads(2)
    names=list(dict.fromkeys(SOURCE_FILES+['tools/regional_visibility170.py','tools/pure_visibility170.py']))
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    for n in names:
        p=OUT/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    (OUT/'protocol.json').write_text(json.dumps(dict(NN_used=False,labels_read_during_inference=False,
        RAW_FPS_measured=False,source_time_horizon_s=.4,regional_seed_recipe='existing165',
        queries=[43.2+i*.2 for i in range(7)],source_sha256=hashes),indent=2),'utf-8')
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0]))
            if p.name=='labels.json' or 'evaluation_data' in p.parts:raise RuntimeError('No labels during inference')
    sys.addaudithook(guard)
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,regional_seeds=True,remote_unpack=True)
    cache=ROOT.parent/'output/native_plane165_20261104/cache_train2_inside';manifest=json.loads((cache/'manifest.json').read_text('utf-8'));models={};rows={}
    for arm in ['legacy','fixed']:
        m=RotorPredictor(no_phase=True,fast_geometry=True,local_phase=True,subpixel_motion=True,entry_init=True,relative_observation=True,relock_support_guard=True,border_motion=True,legacy_visibility=arm=='legacy',gap_prediction='legacy',reference_mode='legacy')
        m.period_worker=worker;m.front.period_estimator=worker;m.front.history_buffer=NativeHistoryBuffer();models[arm]=m;rows[arm]=[]
    try:
        for i,item in enumerate(manifest['frames']):
            t=item['t'];prepared=worker.prepare_uncached(window(cache,t),t)
            for arm,m in models.items():
                worker.cached=(t,prepared);r,frame=m.step(np.empty(0),t);rows[arm].append(r);frame=None
            if i%30==0:print('regional',i,flush=True)
    finally:
        for m in models.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
        prepared=None;worker.cached=None;worker.close()
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    for arm,rs in rows.items():(OUT/f'train2_{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf-8')
    (OUT/'completion.json').write_text(json.dumps(dict(regional_seeds=True,recipe_correction_not_an_algorithm_tuning=True,
        reason='initial common run omitted known regional seeds and detected only6/67; restore165recipe before assessing second recording',
        summary={a:dict(frames=len(rs),detected=sum(r['body'].get('detected',False) for r in rs),valid=sum(r['valid'] for r in rs)) for a,rs in rows.items()},source_sha256=hashes,source_changed=False),indent=2),'utf-8')
