"""Fixed-reference test; one GPU result, no labels during causal inference."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,copy,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_rotor_only import RotorPredictor,SOURCE_FILES,serial
from layout_reference_observer import LayoutReferenceObserver,NativeControlLayoutObserver
from rotor_cuda_worker import CudaRotorFrontWorker
from native_history_buffer import NativeHistoryBuffer
from physical_publication import evaluate
from pure_visibility170 import window
OUT=ROOT.parent/'output/layout176_20261107/integrated'

if __name__=='__main__':
    OUT.mkdir(exist_ok=False);cv2.setNumThreads(2);files=list(dict.fromkeys(SOURCE_FILES+['tools/live_layout176.py','tools/pure_visibility170.py']))
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    for n in files:
        p=OUT/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    (OUT/'protocol.json').write_text(json.dumps(dict(NN_used=False,labels_during_inference=False,RAW_FPS_measured=False,
        arms=dict(control173='173 default',fixed='fixed native output reference and uncertain layout bias',rim_fixed='same with174 native rim recovery',
                  native_control='same candidate front/qualification; layout numbers replaced by native displacement'),
        parameters_searched=False,source_sha256=hashes),indent=2),'utf-8')
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0]))
            if p.name=='labels.json' or 'evaluation_data' in p.parts:raise RuntimeError('No coordinate labels during inference')
    sys.addaudithook(guard);summary={}
    for case,path,start,stop,regional in [('side',ROOT.parent/'output/side_temporal164_20261101/cache_side_warm',110.,114.4,False),
        ('train2',ROOT.parent/'output/native_plane165_20261104/cache_train2_inside',42.5,44.7,True)]:
        worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,regional_seeds=regional,remote_unpack=True)
        models={};rows={'native_control':[]};shadow=NativeControlLayoutObserver();frame=None;prepared=None
        for arm in ['control173','fixed','rim_fixed']:
            m=RotorPredictor(no_phase=True,fast_geometry=True,local_phase=True,subpixel_motion=True,entry_init=True,relative_observation=True,
                relock_support_guard=True,border_motion=True,legacy_visibility=False,gap_prediction='short',reference_mode='legacy' if arm=='control173' else 'fixed')
            m.front.geometry.reconsider_small_cores=arm=='rim_fixed'
            m.period_worker=worker;m.front.period_estimator=worker;m.front.history_buffer=NativeHistoryBuffer();models[arm]=m;rows[arm]=[]
        try:
            manifest=json.loads((path/'manifest.json').read_text('utf-8'))
            for i,item in enumerate(r for r in manifest['frames'] if start<r['t']<=stop):
                t=item['t'];e=window(path,t);prepared=worker.prepare_uncached(e,t)
                for arm,m in models.items():
                    worker.cached=(t,prepared);r,frame=m.step(e[:0],t);rows[arm].append(r);frame=None
                    if arm=='rim_fixed':
                        z=copy.deepcopy(r);meta=z['period_selection'];u=z['force']['input_u']
                        if z['state'].get('two_point_initialization_selected_by_partial_entry'):shadow.two_point_init=True
                        c,v,f,s=shadow.step(t,z['body'],None if u is None else np.asarray(u),.7 if u is not None else 0.,
                            allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,motion=z['motion'])
                        z.update(coefficient=c,valid=v,force=f,state=s);z['physical']=evaluate(z,[.1,.2,.4]) if c else {};rows['native_control'].append(z)
                if i%30==0:print(case,i,flush=True)
        finally:
            for m in models.values():m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
            prepared=None;worker.cached=None;worker.close()
        summary[case]={}
        for arm,rs in rows.items():
            (OUT/f'{case}_{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf-8')
            summary[case][arm]=dict(frames=len(rs),valid=sum(r['valid'] for r in rs),
                geometric_centres=sum(r['structure'].get('current_geometric_center') is not None for r in rs),
                layout_updates=sum(r['state'].get('layout_observation_used',False) for r in rs),
                absolute_reference_changes=sum('measurement_reference_shift_xy' in r['body'] for r in rs))
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    (OUT/'completion.json').write_text(json.dumps(dict(source_changed=False,summary=summary),indent=2),'utf-8');print('COMPLETE',flush=True)
