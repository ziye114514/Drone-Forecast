"""Freeze inverse320/289 mechanics before opening recording4 RAW.

One existing rotor-only front, three force continuation outputs, same native
events/current state. Outline references are extracted ONLY after inference,
then require visual review. No independent COM/attitude/RPM truth claim.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_rotor_only import SOURCE_FILES,serial
from tools.check_inverse322 import model
from inverse_normal_response import InverseNormalResponse
from tangential_response_memory import adapt
from rotor_cuda_worker import CudaRotorFrontWorker
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from live_decode import native_batches
from pure_visibility170 import window
from reference_boxes181 import outline_box
from physical_publication import evaluate
from score_rotor164 import score

if __name__=='__main__':
    out=ROOT.parent/'output/frozen324_20261109';out.mkdir(exist_ok=False)
    files=sorted(set(SOURCE_FILES+['tools/frozen_inverse324.py','tools/check_inverse322.py']));hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files if (ROOT/f).is_file()}
    protocol=dict(raw='E:/dataset/experiment/4/media/event.raw',interval=[10.,16.],source_sha256=hashes,
        source_frozen_before_raw_open=True,no_recording_specific_rules=True,no_parameter_search=True,
        no_future_reference_during_inference=True,NN_used=False,GPU_used=True,RAW_FPS_measured=False,
        historical_project_may_have_used_recording_not_independent_blind_test=True,
        references_are_weak_event_visible_outline_not_COM=True,fixed25_origins='10.8..15.6 step0.2',horizon_s=.4)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in hashes:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True,regional_graph=True)
    baseline=model(worker);inverse=InverseNormalResponse();outputs={a:[] for a in ('control','inverse320','combined')};cache=out/'cache';cache.mkdir();manifest=dict(spec=dict(start=10.),frames=[])
    try:
        for i,(t,e) in enumerate(native_batches(Path(protocol['raw']),10.,6.,False,33333,reader_backend='cooperative')):
            file=f'{i:04d}.npy';np.save(cache/file,e);manifest['frames'].append(dict(t=t,file=file))
            worker.cached=(t,worker.prepare_uncached(e,t));old,frame=baseline.step(e[:0],t);frame=None
            r=inverse.step(old);combined=adapt(r,inverse.last_tangent_info)
            for a,p in dict(control=old,inverse320=r,combined=combined).items():
                p['physical']=evaluate(p,[.1,.2,.4]) if p['coefficient'] else {};outputs[a].append(p)
                assert p['coefficient']==old['coefficient'];assert p['valid']==old['valid']
            if i%30==0:print(i,t,'native detected',old['body'].get('detected'),'valid',old['valid'],'inverse',r['inverse_normal_response'].get('external_partition_observable'),flush=True)
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    finally:
        baseline.front.previous=None;baseline.front.geometry.count_current=None;baseline.period_worker=None;worker.cached=None;worker.close();configure_native(None)
    (cache/'manifest.json').write_text(json.dumps(manifest),'utf8')
    for a,rr in outputs.items():(out/f'{a}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rr),'utf8')
    (out/'inference_complete.json').write_text(json.dumps(dict(frames=len(manifest['frames']),source_hashes_verified=True,original_raw_unchanged=True)),'utf8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records={};images=[]
    for t in np.round(np.arange(10.8,16.0001,.2),1):
        e=window(cache,t);e=e[e['t']>=round(t*1e6)-20000];ids=e['y'].astype(np.int64)*1280+e['x'];count=np.bincount(ids,minlength=1280*720).reshape(720,1280)
        b=outline_box(count,8);weak=outline_box(count,4);delta=None if b is None or weak is None else float(np.linalg.norm((np.array(b[:2])+b[2:]-np.array(weak[:2])-weak[2:])/2))
        records[f'{t:.1f}']=dict(t=float(t),state='present' if b is not None else 'absent',box=b,weaker_outline_box=weak,threshold_center_disagreement_px=delta)
        im=np.uint8(255-np.clip(count*20,0,255));im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        if b is not None:cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,160,0),2)
        # Full frame retained. A tight crop could hide a false background box.
        images.append((t,cv2.cvtColor(im,cv2.COLOR_BGR2RGB),delta))
    ref=out/'references.json';ref.write_text(json.dumps(dict(records=records,assistant_visual_review_pending=True,
        prediction_used_to_generate_references=False,reference_method='all native pixels,count8/count4,components>=16',independent_human_COM_truth=False),indent=2),'utf8')
    for page,start in enumerate(range(0,len(images),9)):
        fig,axes=plt.subplots(3,3,figsize=(15,9));[ax.axis('off') for ax in axes.flat]
        for ax,(t,im,d) in zip(axes.flat,images[start:start+9]):ax.imshow(im);ax.set_title(f'{t:.1f}s outline disagreement {d:.1f}px' if d is not None else f'{t:.1f}s no dense outline')
        fig.tight_layout();fig.savefig(out/f'review{page}.png',dpi=120);plt.close(fig)
    origins=np.round(np.arange(10.8,15.6001,.2),1);scores={a:score(out/f'{a}.jsonl',ref,origins) for a in outputs}
    result=dict(cases=scores,protocol=protocol,reference_visual_review_pending=True,
        native_detected_frames=sum(bool(r['body'].get('detected')) for r in outputs['control']),native_valid_frames=sum(r['valid'] for r in outputs['control']))
    (out/'results.json').write_text(json.dumps(result,indent=2),'utf8');print({a:s['error'] for a,s in scores.items()},flush=True)
