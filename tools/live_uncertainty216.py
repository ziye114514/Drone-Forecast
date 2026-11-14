"""Frozen public-factory test: same native observations, three force arms."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,argparse,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from run_rotor_only import RotorPredictor,SOURCE_FILES,serial
from native_arc_search import NativeArcSearch
from batched_arc_fit import configure_native
import cProfile,pstats,io
from rotor_cuda_worker import CudaRotorFrontWorker
from native_history_buffer import NativeHistoryBuffer
from pure_visibility170 import window
BASE=ROOT.parent/'output/uncertainty216_20261108'
FILES=list(dict.fromkeys(SOURCE_FILES+['tools/live_uncertainty216.py']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',choices=['13'],required=True);args=p.parse_args()
    out=BASE/f'recording{args.recording}';out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(2)
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in FILES}
    for n in FILES:
        dst=out/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/n).read_bytes())
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,labels_during_inference=False,NN_used=False,
        thresholds_fitted_to_recordings=False,shared_native_gpu_observations=True,RAW_FPS_measured=False,
        changes='Fast hidden normal amplitude, smooth external residual and its rate; isotropic control has same fast-noise trace'),indent=2),'utf-8')
    def guard(event,items):
        if event=='open' and items and isinstance(items[0],(str,bytes,os.PathLike)):
            pp=Path(os.fsdecode(items[0]))
            if pp.name in ('labels.json','references.json') or 'evaluation_data' in pp.parts:raise RuntimeError('No labels during inference')
    sys.addaudithook(guard)
    parent='motion181_20261107' if args.recording=='15' else 'phase185_20261107'
    cache=ROOT.parent/f'output/{parent}/cache_recording{args.recording}'
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    configure_native(NativeArcSearch(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    profile=cProfile.Profile();models={};rows={};prepared=None;frame=None
    for arm in ('regional_adaptive',):
        m=RotorPredictor(no_phase=True,fast_geometry=True,local_phase=True,subpixel_motion=True,entry_init=True,
            relative_observation=True,relock_support_guard=True,border_motion=True,gap_prediction='short',reference_mode='fixed',
            unknown_force_model='plane-adaptive')
        m.front.current_region_search=True
        m.period_worker=worker;m.front.period_estimator=worker;m.front.history_buffer=NativeHistoryBuffer();models[arm]=m;rows[arm]=[]
    try:
        manifest=json.loads((cache/'manifest.json').read_text('utf-8'))
        for i,item in enumerate(manifest['frames']):
            t=item['t'];events=window(cache,t);prepared=worker.prepare_uncached(events,t)
            for arm,m in models.items():
                worker.cached=(t,prepared)
                if args.recording=='13':profile.enable()
                r,frame=m.step(events[:0],t)
                profile.disable();frame=None;rows[arm].append(r)
                if (r.get('regional_graph_fallback') or {}).get('used'):assert r['force'].get('input_u') is None
            if i%45==0:print(args.recording,i,rows['regional_adaptive'][-1]['valid'],flush=True)
    finally:
        configure_native(None)
        for m in models.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
        prepared=None;worker.cached=None;worker.close()
    buf=io.StringIO()
    if args.recording=='13':pstats.Stats(profile,stream=buf).sort_stats('cumtime').print_stats(35)
    (out/'profile.txt').write_text(buf.getvalue(),'utf-8')
    summary={}
    for arm,rr in rows.items():
        (out/f'{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rr),'utf-8')
        summary[arm]=dict(frames=len(rr),valid=sum(r['valid'] for r in rr),regions=sum(r['body'].get('detected',False) for r in rr),recovered=sum(bool((r.get('regional_graph_fallback') or {}).get('used')) for r in rr))
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    (out/'completion.json').write_text(json.dumps(dict(source_changed=False,summary=summary),indent=2),'utf-8');print(json.dumps(summary),flush=True)
