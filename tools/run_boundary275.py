"""Actual rotor-only front: rank admissible split pairs by measured boundary.

One frozen selector, shared current GPU packets, original algorithm control.
No motion/force parameter tuning and no labels until inference completes.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,argparse,hashlib,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from audit_continuity266 import model
from run_rotor_only import serial,SOURCE_FILES
from rotor_cuda_worker import CudaRotorFrontWorker
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window
from replay_joint236 import source
from score_rotor164 import score


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',choices=['13','15','16'],required=True);p.add_argument('--attempt',default='');args=p.parse_args();n=args.recording
    OUT=ROOT.parent/f'output/boundary275_20261108/recording{n}{args.attempt}';OUT.mkdir(parents=True,exist_ok=False)
    files=sorted(set(list(SOURCE_FILES)+['tools/run_boundary275.py','curvature_visibility.py','fast_structure_front.py']))
    files=[f for f in files if (ROOT/f).is_file()];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(recording=n,source_sha256=hashes,change='only admissible original-arc pair ranking: capped boundary residual instead of enclosed foreground area',
        no_threshold_or_force_tuning=True,no_recording_specific_rules=True,NN_used=False,GPU_used=True,
        heldout_experiment1_opened=False,shared_GPU_features=True,RAW_FPS_measured=False,labels_read_during_inference=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        d=OUT/'source_snapshot'/f;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes((ROOT/f).read_bytes())
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    models={arm:model(worker) for arm in ['control','boundary']};models['boundary'].front.geometry.arc_pair_selection='boundary_likelihood'
    rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()];outputs={arm:[] for arm in models};maxdelta=0.;changes=[]
    cache=ROOT.parent/f'output/{"motion181_20261107" if n=="15" else "phase185_20261107"}/cache_recording{n}'
    try:
        for i,old in enumerate(rows):
            t=old['t'];events=window(cache,t);features=worker.prepare_uncached(events,t);frames={}
            for arm,m in models.items():
                worker.cached=(t,features);r,frame=m.step(events[:0],t);outputs[arm].append(r);frames[arm]=r
                if arm=='control' and r['coefficient']:
                    for h,pred in r['physical'].items():maxdelta=max(maxdelta,float(np.max(abs(np.asarray(pred)-old['physical'][h]))))
                frame=None
            a,b=frames['control'],frames['boundary']
            if json.dumps(a['geometry'].get('rotors'),default=serial,sort_keys=True)!=json.dumps(b['geometry'].get('rotors'),default=serial,sort_keys=True):
                changes.append(dict(t=t,index=i,control_count=len(a['geometry'].get('rotors',[])),candidate_count=len(b['geometry'].get('rotors',[]))))
                if len(changes)<=8:
                    e=events[events['t']>=round(t*1e6)-20000];ids=e['y'].astype(np.int64)*1280+e['x'];counts=np.bincount(ids,minlength=720*1280).reshape(720,1280)
                    np.savez_compressed(OUT/f'review_{i:04d}.npz',counts=counts)
            if i%30==0:print(n,i,'control delta',maxdelta,'changed',len(changes),flush=True)
        assert maxdelta<1e-6,maxdelta
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    finally:
        for m in models.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
        worker.cached=None;worker.close();configure_native(None)
    for arm,rs in outputs.items():(OUT/f'{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf8')
    folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
    ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
    scores={arm:score(OUT/f'{arm}.jsonl',ref,origins) for arm in outputs}
    (OUT/'results.json').write_text(json.dumps(dict(scores=scores,control_max_px=maxdelta,changed_frames=changes,protocol=protocol),indent=2),'utf8')
    print(n,{arm:s['error'] for arm,s in scores.items()},flush=True)
