"""First use of reserved experiment1, frozen geometry276, no tuning allowed."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from physical_geometry_refinement import PhysicalGeometryRefinement
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
    OUT=ROOT.parent/'output/frozen_unseen278_20261108';OUT.mkdir(exist_ok=False)
    files=sorted(set(list(SOURCE_FILES)+['tools/frozen_unseen278.py','physical_geometry_refinement.py','metric_model_relock.py','native_event_alignment.py']))
    files=[f for f in files if (ROOT/f).is_file()];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(raw='E:/dataset/experiment/1/media/event.raw',interval=[10.,16.],source_sha256=hashes,
        first_use_of_reserved_recording=True,previously_developed_recordings=['13','15','16','11'],
        no_recording_specific_rules=True,no_parameter_search=True,NN_used=False,GPU_used=True,
        identical_geometry276_code=True,reference_method='native20ms,count8/count4,componentarea16',
        fixed_origins='10.8..15.6 step0.2,25queries',horizon_s=.4,
        no_labels_or_reference_extracted_until_inference_complete=True,
        RAW_FPS_measured=False,not_independent_COM_truth=True,default_changed=False,
        frozen_before_first_raw_read=True,failed_result_will_not_be_tuned_on_this_recording=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    header=[]
    with Path(protocol['raw']).open('rb') as stream:
        for _ in range(32):
            b=stream.readline(1024)
            if not b.startswith(b'%'):break
            header.append(b.decode('ascii','replace').strip())
            if b.strip()==b'% end':break
    (OUT/'raw_header.json').write_text(json.dumps(header,indent=2),'utf8')
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    control=model(worker);refined=PhysicalGeometryRefinement(model(worker));outputs={k:[] for k in ['control','refined']}
    cache=OUT/'cache';cache.mkdir();manifest=dict(spec=dict(start=10.),frames=[]);image_times=np.round(np.arange(10.8,16.0001,.2),1)
    try:
        for i,(t,e) in enumerate(native_batches(Path(protocol['raw']),10.,6.,False,33333,reader_backend='cooperative')):
            name=f'{i:04d}.npy';np.save(cache/name,e);manifest['frames'].append(dict(t=t,file=name))
            features=worker.prepare_uncached(e,t)
            for arm,m in [('control',control),('refined',refined)]:
                worker.cached=(t,features);r,frame=m.step(e,t);outputs[arm].append(r);frame=None
            if i%30==0:print(i,t,'current target',outputs['control'][-1]['body'].get('detected'),'valid',outputs['control'][-1]['valid'],flush=True)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    finally:
        for m in [control,refined.predictor]:m.front.previous=None;m.front.geometry.count_current=None;m.period_worker=None
        worker.cached=None;worker.close();configure_native(None)
    (OUT/'inference_complete.json').write_text(json.dumps(dict(frames=len(manifest['frames']),source_unchanged=True)),'utf8')
    (cache/'manifest.json').write_text(json.dumps(manifest),'utf8')
    for arm,rs in outputs.items():(OUT/f'{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rs),'utf8')
    records={};review=[]
    for t in image_times:
        e=window(cache,t);e=e[e['t']>=round(t*1e6)-20000]
        ids=e['y'].astype(np.int64)*1280+e['x'];count=np.bincount(ids,minlength=720*1280).reshape(720,1280)
        b=outline_box(count,8);w=outline_box(count,4)
        disagreement=None if b is None or w is None else float(np.linalg.norm((np.array(b[:2])+b[2:]-np.array(w[:2])-w[2:])/2))
        records[f'{t:.1f}']=dict(t=float(t),state='present' if b is not None else 'absent',box=b,weaker_outline_box=w,
            threshold_center_disagreement_px=disagreement,reference_is_independent_human_GT=False)
        im=cv2.cvtColor(np.uint8(255-np.clip(count*20,0,255)),cv2.COLOR_GRAY2BGR)
        if b is not None:
            cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,150,0),1)
            lo=np.maximum(0,np.array(b[:2])-30);hi=np.minimum([1280,720],np.array(b[2:])+30)
            im=im[lo[1]:hi[1],lo[0]:hi[0]]
        review.append((t,im,disagreement))
    refs=OUT/'references.json';refs.write_text(json.dumps(dict(records=records,
        scope='fixed native visible-outline centre, not COM or humanGT',assistant_visual_review_pending=True),indent=2),'utf8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for page,start in enumerate(range(0,len(review),9)):
        fig,axes=plt.subplots(3,3,figsize=(12,9))
        for ax in axes.flat:ax.axis('off')
        for ax,(t,im,d) in zip(axes.flat,review[start:start+9]):
            ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(f'{t:.1f}s d={d:.1f}px' if d is not None else f'{t:.1f}s no dense outline')
        fig.tight_layout();fig.savefig(OUT/f'review{page}.png',dpi=120);plt.close(fig)
    origins=np.round(np.arange(10.8,15.6001,.2),1)
    results={arm:score(OUT/f'{arm}.jsonl',refs,origins) for arm in outputs}
    diagnostics={arm:dict(frames=len(rs),current_target=sum(r['body'].get('detected',False) for r in rs),
        valid=sum(r['valid'] for r in rs)) for arm,rs in outputs.items()}
    (OUT/'results.json').write_text(json.dumps(dict(scores=results,diagnostics=diagnostics,protocol=protocol,
        reference_visual_review_pending=True),indent=2),'utf8')
    print({k:v['error'] for k,v in results.items()},diagnostics,flush=True)
