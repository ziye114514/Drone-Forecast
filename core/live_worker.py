"""Current-frame FP32-I/O CUDA worker without historical benchmark imports."""
from pathlib import Path
import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from multiprocessing import shared_memory

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'))
os.environ['YOLO_OFFLINE']='true'
os.environ['YOLO_AUTOINSTALL']='false'


def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.name in ['inputs.json','targets.json','detections.jsonl','geometry.jsonl'] or path.is_relative_to((ROOT.parent/'output/raw_only_project/evaluation_data').resolve()):
            raise RuntimeError('RAW worker cannot read dataset cache/truth: '+str(path))


def main(name,weights,expected_sha256,cache_dir,body_association=False,frame_rate=25.,center_ensemble_weights=None,center_ensemble_sha256=None,weak_verifier_weights=None,weak_verifier_sha256=None,weak_verifier_max_side=192.,current_box_gate=False):
    cache=Path(cache_dir).resolve();cache.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(cache/'ultralytics')
    os.environ['CUDA_CACHE_PATH']=str(cache/'cuda')
    sys.addaudithook(guard);began=time.perf_counter()
    model=Path(weights).resolve()
    assert hashlib.sha256(model.read_bytes()).hexdigest()==expected_sha256
    import cv2,numpy as np,torch
    torch.set_num_threads(2)
    from event_drone_forecast import detector_backend as backend_module
    with contextlib.redirect_stdout(sys.stderr):
        backend=backend_module.Backend('graph',model_path=model)
        secondary=None
        if center_ensemble_weights is not None:
            path=Path(center_ensemble_weights).resolve()
            assert center_ensemble_sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest()==center_ensemble_sha256
            secondary=backend_module.Backend('graph',model_path=path)
        verifier=None
        if weak_verifier_weights is not None:
            path=Path(weak_verifier_weights).resolve()
            assert weak_verifier_sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest()==weak_verifier_sha256
            verifier=backend_module.Backend('graph',model_path=path)
        associator=None
        if body_association:
            from body_association import BodyAssociation
            associator=BodyAssociation(frame_rate)
        gate=None
        if current_box_gate:
            from current_box_gate import CurrentBoxGate
            gate=CurrentBoxGate()
        # The usual all-background warmups take the empty-NMS path. Warm the
        # actual nonempty CPU postprocess before the first source deadline.
        postprocess_warm_start=time.perf_counter()
        synthetic=np.zeros((1,5,5040),np.float32)
        synthetic[0,:4,0]=[200.,200.,50.,50.];synthetic[0,4,0]=.9
        synthetic[0,:4,1]=[201.,200.,50.,50.];synthetic[0,4,1]=.8
        synthetic[0,:4,2]=[500.,200.,50.,50.];synthetic[0,4,2]=.7
        # One box bypasses actual overlap suppression in optimized NMS.
        assert len(backend_module.boxes(synthetic))==2
        postprocess_warm_ms=(time.perf_counter()-postprocess_warm_start)*1000
    shared=shared_memory.SharedMemory(name=name,create=False)
    image=np.ndarray((720,1280,3),np.uint8,buffer=shared.buf);image.setflags(write=False);counter=0
    native_image=None
    if shared.size>=2*720*1280*3:
        native_image=np.ndarray((720,1280,3),np.uint8,buffer=shared.buf,offset=720*1280*3)
        native_image.setflags(write=False)
    try:
        print(json.dumps(dict(ready=True,weight_sha256=expected_sha256,precision=32,
            precision_scope='FP32 input/output; internal arithmetic follows the hashed ONNX graph',
            internal_precision_hint=backend.session.get_modelmeta().custom_metadata_map.get('edf.internal_precision','not_declared'),
            cold_start_s=time.perf_counter()-began,provider_options=backend.provider_options,
            nonempty_postprocess_warmup_ms=postprocess_warm_ms,
            center_ensemble_weight_sha256=center_ensemble_sha256,
            weak_verifier_weight_sha256=weak_verifier_sha256,
            body_association=associator.provenance if associator else None)),flush=True)
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('stop'):break
            assert request['id']==counter;counter+=1
            frame_sha=hashlib.sha256(image.tobytes()).hexdigest() if request.get('verify_frame') else None
            tick=time.perf_counter();network_input=backend_module.preprocess(image)
            raw=backend.run(network_input);assert np.isfinite(raw).all()
            weak_hint=request.get('weak_continuation_hint')
            b=backend_module.boxes(raw.copy() if associator is not None or weak_hint is not None else raw)
            elapsed=(time.perf_counter()-tick)*1000
            boxes=[dict(bbox=r[:4].tolist(),confidence=float(r[4]),class_id=int(r[5])) for r in b]
            extra={}
            if request.get('native_window_fallback') and not request.get('warmup'):
                audit=dict(attempted=not bool(boxes),accepted=False,original_full_frame_preserved=bool(boxes))
                if not boxes:
                    if native_image is None:raise RuntimeError('Missing second current-frame buffer')
                    expected=request.get('native_frame_sha256')
                    if expected and hashlib.sha256(native_image.tobytes()).hexdigest()!=expected:
                        raise RuntimeError('Native window recovery frame mismatch')
                    start=time.perf_counter()
                    native_raw=backend.run(backend_module.preprocess(native_image))
                    recovered=backend_module.boxes(native_raw)
                    boxes=[dict(bbox=r[:4].tolist(),confidence=float(r[4]),class_id=int(r[5]),
                        native_window_recovered=True,current_detection=True) for r in recovered]
                    audit.update(accepted=bool(boxes),inference_ms=(time.perf_counter()-start)*1000,
                        source_t=request['t'],uses_no_future_events=True,confidence_threshold=.25)
                    elapsed+=(time.perf_counter()-start)*1000
                extra['native_window_recovery']=audit
            if weak_hint is not None and not boxes and not request.get('warmup'):
                # ByteTrack-style low-score continuation, using this same
                # current NN output. The hint is search support, never a box
                # measurement; no new inference or confidence promotion.
                from live_roi_refine import select_refinement
                began_weak=time.perf_counter()
                candidates=backend_module.boxes(raw.copy(),confidence=.1)
                selected,iou=select_refinement(weak_hint,candidates,np.zeros(2),1.,minimum_confidence=.1)
                if selected is not None:
                    boxes=[dict(bbox=selected[:4].tolist(),confidence=float(selected[4]),class_id=int(selected[5]),
                        weak_continuation=True,low_confidence=True,search_hint_is_not_a_measurement=True)]
                extra['weak_body_continuation']=dict(attempted=True,accepted=selected is not None,
                    candidates=len(candidates),selected_IoU=iou,current_NN_forward_passes_added=0,
                    confidence_promoted=False,CPU_ms=(time.perf_counter()-began_weak)*1000)
            if verifier is not None and not request.get('warmup'):
                reference=max(boxes,key=lambda row:row['confidence']) if boxes else None
                hint=request.get('weak_verifier_hint')
                current_parent=reference is not None
                candidate=reference['bbox'] if current_parent else hint
                eligible=candidate is not None and max(np.subtract(candidate[2:],candidate[:2]))<=weak_verifier_max_side
                eligible=bool(eligible and (not current_parent or reference['confidence']<.45))
                if eligible:
                    from live_roi_refine import select_refinement
                    began_verify=time.perf_counter();answer=backend_module.boxes(verifier.run(network_input))
                    best,iou=select_refinement(candidate,answer,np.zeros(2),1.)
                    if best is not None:
                        if current_parent:
                            boxes=[dict(reference,confidence=float(best[4]),cross_model_verified=True,
                                original_full_frame_confidence=reference['confidence'],verification_bbox=best[:4].tolist())]
                        else:
                            boxes=[dict(bbox=best[:4].tolist(),confidence=float(best[4]),class_id=int(best[5]),
                                cross_model_recovered=True,search_hint_is_not_measurement=True)]
                    extra['weak_verifier']=dict(attempted=True,accepted=best is not None,
                        parent_current_detection=current_parent,selected_IoU=iou,candidates=len(answer),
                        inference_ms=(time.perf_counter()-began_verify)*1000,
                        parent_localization_preserved=current_parent,current_frame_only=True,
                        unrelated_adapter_detections_not_used=True)
            if request.get('verify_weak_roi') and not request.get('warmup') and boxes:
                # A second current image measurement, never confidence
                # promotion from persistence alone. Reuse the existing crop
                # transform and strict spatial agreement gates.
                reference=max(boxes,key=lambda row:row['confidence'])
                size=np.subtract(reference['bbox'][2:],reference['bbox'][:2])
                eligible=reference['confidence']<.45 and max(size)<=192.
                if eligible:
                    from live_roi_refine import refine
                    began_verify=time.perf_counter()
                    verified,audit=refine(backend,backend_module,image,reference['bbox'])
                    if verified is not None:
                        # Preserve native full-frame localization; the crop
                        # verifies semantics and is not a new center gauge.
                        boxes=[dict(reference,confidence=verified['confidence'],weak_roi_verified=True,
                            verification_bbox=verified['bbox'],
                            original_full_frame_confidence=reference['confidence'])]
                    extra['weak_roi_verification']=dict(audit,accepted=verified is not None,
                        verification_ms=(time.perf_counter()-began_verify)*1000,
                        current_crop_measurement=True,source='one_current_crop_of_weak_small_detection')
            if request.get('recover_roi') and not request.get('warmup') and not boxes:
                from live_roi_refine import refine
                hint=request.get('roi_hint');recovery_start=time.perf_counter()
                recovered=None;recovery=dict(reason='no_recent_strong_ROI')
                if hint is not None:
                    recovered,recovery=refine(backend,backend_module,image,hint)
                if recovered is not None:
                    boxes=[dict(recovered,roi_recovered=True,current_detection=True,
                        search_hint_is_not_a_measurement=True)]
                extra.update(roi_recovery=dict(recovery,attempted=hint is not None,accepted=recovered is not None,
                    recovery_ms=(time.perf_counter()-recovery_start)*1000,
                    source='current_frame_zoom_NN_after_full_frame_miss'))
            if secondary is not None:
                second_start=time.perf_counter()
                second=backend_module.boxes(secondary.run(backend_module.preprocess(image)))
                extra.update(ensemble_boxes=[dict(bbox=row[:4].tolist(),confidence=float(row[4]),class_id=int(row[5])) for row in second],
                    ensemble_inference_ms=(time.perf_counter()-second_start)*1000)
            if request.get('refine_roi') and not request.get('warmup'):
                from live_roi_refine import refine
                ref_start=time.perf_counter()
                if boxes:
                    reference=max(boxes,key=lambda row:row['confidence'])
                    refined,audit=refine(backend,backend_module,image,reference['bbox'])
                else:refined,audit=None,dict(reason='no_current_full_frame_detection')
                extra.update(refined_body=refined,refinement=audit,refinement_ms=(time.perf_counter()-ref_start)*1000)
            if associator is not None and not request.get('warmup'):
                # NMS mutates xywh, so each confidence pass gets its own copy.
                candidate=backend_module.boxes(raw.copy(),confidence=.1)
                high=candidate[candidate[:,4]>=.25]
                if high.shape!=b.shape or not np.allclose(high,b,atol=1e-5,rtol=0):
                    raise RuntimeError('High-score NMS baseline changed')
                selected,audit=associator.update(candidate,float(request['t']))
                extra=dict(candidates=[dict(bbox=r[:4].tolist(),confidence=float(r[4]),class_id=int(r[5])) for r in candidate],
                           associated_boxes=selected,association=audit,high_score_invariant=True)
            if gate is not None and not request.get('warmup'):
                boxes,audit=gate.step(boxes,float(request['t']),request.get('current_association_hint'))
                extra['current_box_gate']=audit
            print(json.dumps(dict(id=request['id'],boxes=boxes,inference_ms=elapsed,
                frame_sha256=frame_sha,**extra),allow_nan=False),flush=True)
    finally:
        image=None;native_image=None;shared.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shared-memory',required=True)
    p.add_argument('--precision',type=int,choices=[32],default=32)
    p.add_argument('--weights',required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--cache-dir',required=True)
    p.add_argument('--body-association',action='store_true')
    p.add_argument('--frame-rate',type=float,default=25.)
    p.add_argument('--center-ensemble-weights');p.add_argument('--center-ensemble-sha256')
    p.add_argument('--weak-verifier-weights');p.add_argument('--weak-verifier-sha256')
    p.add_argument('--weak-verifier-max-side',type=float,default=192.)
    p.add_argument('--current-box-gate',action='store_true')
    a=p.parse_args()
    if a.weak_verifier_weights and (a.body_association or a.center_ensemble_weights):p.error('Weak verifier is a separate current-detection mode')
    main(a.shared_memory,a.weights,a.expected_sha256,a.cache_dir,a.body_association,a.frame_rate,a.center_ensemble_weights,a.center_ensemble_sha256,a.weak_verifier_weights,a.weak_verifier_sha256,a.weak_verifier_max_side,a.current_box_gate)
