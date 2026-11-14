"""Frozen new recording3 test of tangential unknown-force memory286."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tangential_response_memory import TangentialResponseMemory,adapt
from physical_publication import evaluate
from audit_continuity266 import model
from run_rotor_only import serial,SOURCE_FILES
from rotor_cuda_worker import CudaRotorFrontWorker
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from live_decode import native_batches
from pure_visibility170 import window
from reference_boxes181 import outline_box
from score_rotor164 import score

if __name__=='__main__':
    OUT=ROOT.parent/'output/frozen_tangent287_20261108';OUT.mkdir(exist_ok=False)
    files=sorted(set(list(SOURCE_FILES)+['tools/frozen_tangent287.py','tangential_response_memory.py']))
    files=[f for f in files if (ROOT/f).is_file()];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(raw='E:/dataset/experiment/3/media/event.raw',interval=[10.,16.],source_sha256=hashes,
        source_frozen_before_raw_open=True,no_recording_specific_rules=True,no_parameter_search=True,
        tested_candidate_is_tangent286_only=True,geometry276_and_tube_recovery_not_used=True,
        no_reference_extraction_until_inference_complete=True,reference_method='native20ms,count8/count4,components>=16',
        evaluation='fixed25origins10.8..15.6 step0.2,0.4s',NN_used=False,GPU_used=True,RAW_FPS_measured=False,
        not_independent_COM_truth=True,do_not_retune_after_test=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    baseline=model(worker);memory=TangentialResponseMemory();outputs={k:[] for k in ['control','tangent']};cache=OUT/'cache';cache.mkdir();manifest=dict(spec=dict(start=10.),frames=[])
    try:
        for i,(t,e) in enumerate(native_batches(Path(protocol['raw']),10.,6.,False,33333,reader_backend='cooperative')):
            file=f'{i:04d}.npy';np.save(cache/file,e);manifest['frames'].append(dict(t=t,file=file))
            features=worker.prepare_uncached(e,t);worker.cached=(t,features);r,frame=baseline.step(e[:0],t);frame=None
            outputs['control'].append(r);info=memory.observe(r);s=adapt(r,info);s['physical']=evaluate(s,[.1,.2,.4]) if s['coefficient'] else {};outputs['tangent'].append(s)
            if i%30==0:print(i,t,'target',r['body'].get('detected'),'valid',r['valid'],'memory',info.get('qualified'),flush=True)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    finally:
        baseline.front.previous=None;baseline.front.geometry.count_current=None;baseline.period_worker=None;worker.cached=None;worker.close();configure_native(None)
    (cache/'manifest.json').write_text(json.dumps(manifest),'utf8');(OUT/'inference_complete.json').write_text(json.dumps(dict(frames=len(manifest['frames']),source_unchanged=True)),'utf8')
    for arm,rs in outputs.items():(OUT/f'{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf8')
    refs={};images=[]
    for t in np.round(np.arange(10.8,16.0001,.2),1):
        e=window(cache,t);e=e[e['t']>=round(t*1e6)-20000];ids=e['y'].astype(np.int64)*1280+e['x'];count=np.bincount(ids,minlength=1280*720).reshape(720,1280)
        b=outline_box(count,8);weak=outline_box(count,4);delta=None if b is None or weak is None else float(np.linalg.norm((np.array(b[:2])+b[2:]-np.array(weak[:2])-weak[2:])/2))
        refs[f'{t:.1f}']=dict(t=float(t),state='present' if b is not None else 'absent',box=b,weaker_outline_box=weak,threshold_center_disagreement_px=delta)
        im=cv2.cvtColor(np.uint8(255-np.clip(count*20,0,255)),cv2.COLOR_GRAY2BGR)
        if b is not None:
            cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,140,0),1);lo=np.maximum(0,np.array(b[:2])-30);hi=np.minimum([1280,720],np.array(b[2:])+30);im=im[lo[1]:hi[1],lo[0]:hi[0]]
        images.append((t,im,delta))
    refpath=OUT/'references.json';refpath.write_text(json.dumps(dict(records=refs,assistant_visual_review_pending=True,scope='fixed visible-outline centre, not independentCOM truth'),indent=2),'utf8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for page,start in enumerate(range(0,len(images),9)):
        fig,axes=plt.subplots(3,3,figsize=(12,9))
        for ax in axes.flat:ax.axis('off')
        for ax,(t,im,d) in zip(axes.flat,images[start:start+9]):
            ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(f'{t:.1f}s d={d:.1f}px' if d is not None else f'{t:.1f}s no outline')
        fig.tight_layout();fig.savefig(OUT/f'review{page}.png',dpi=120);plt.close(fig)
    origins=np.round(np.arange(10.8,15.6001,.2),1);scores={arm:score(OUT/f'{arm}.jsonl',refpath,origins) for arm in outputs}
    (OUT/'results.json').write_text(json.dumps(dict(scores=scores,protocol=protocol,reference_visual_review_pending=True),indent=2),'utf8')
    print({k:v['error'] for k,v in scores.items()},flush=True)
