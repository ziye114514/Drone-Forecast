"""Owned shared-memory image, current-frame-only local CUDA graph worker."""
from pathlib import Path
import argparse,contextlib,hashlib,json,os,sys,time
from multiprocessing import shared_memory
ROOT=Path(__file__).resolve().parents[2]
os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_CONFIG_DIR']=str(ROOT.parent/'tmp/yolo_config44');os.environ['CUDA_CACHE_PATH']=str(ROOT.parent/'tmp/cuda_cache44')
MODEL32=ROOT.parent/'output/stage44_onnx_backend_20261019/weights/best.onnx';SHA32='e2531556ad0431d904fb0d1d432727a2e2dd9eaed9e098a6ed3d9e55f474db0b'
MODEL16=ROOT.parent/'output/stage46_fp16_graph_20261019/weights/best_fp16.onnx';SHA16='d440b0b44590acfaa56836bcc9d88c5741acc8b6617c960fd24fab481067c049'
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.name in ['inputs.json','targets.json','detections.jsonl','geometry.jsonl'] or path.is_relative_to((ROOT.parent/'output/raw_only_project/evaluation_data').resolve()):raise RuntimeError('RAW worker cannot read dataset cache/truth: '+str(path))
def main(name,precision,weights=None,expected_sha256=None,cache_dir=None,body_association=False,frame_rate=25.):
    if cache_dir is not None:
        cache=Path(cache_dir).resolve();cache.mkdir(parents=True,exist_ok=True)
        os.environ['YOLO_CONFIG_DIR']=str(cache/'ultralytics');os.environ['CUDA_CACHE_PATH']=str(cache/'cuda')
    sys.addaudithook(guard);began=time.perf_counter();model=Path(weights).resolve() if weights else (MODEL16 if precision==16 else MODEL32);expected=expected_sha256 or (SHA16 if precision==16 else SHA32);assert hashlib.sha256(model.read_bytes()).hexdigest()==expected
    import cv2,numpy as np,torch
    torch.set_num_threads(2);sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage45'))
    import graph45
    with contextlib.redirect_stdout(sys.stderr):
        backend=graph45.Backend('graph',model_path=model)
        associator=None
        if body_association:
            sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
            from body_association import BodyAssociation
            associator=BodyAssociation(frame_rate)
    shared=shared_memory.SharedMemory(name=name,create=False);image=np.ndarray((720,1280,3),np.uint8,buffer=shared.buf);image.setflags(write=False);counter=0
    try:
        print(json.dumps(dict(ready=True,weight_sha256=expected,precision=precision,cold_start_s=time.perf_counter()-began,provider_options=backend.provider_options,body_association=associator.provenance if associator else None)),flush=True)
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('stop'):break
            assert request['id']==counter;counter+=1
            frame_sha=hashlib.sha256(image.tobytes()).hexdigest() if request.get('verify_frame') else None
            tick=time.perf_counter();raw=backend.run(graph45.preprocess(image));assert np.isfinite(raw).all();b=graph45.boxes(raw.copy() if associator is not None else raw);elapsed=(time.perf_counter()-tick)*1000
            boxes=[dict(bbox=r[:4].tolist(),confidence=float(r[4]),class_id=int(r[5])) for r in b]
            extra={}
            if associator is not None and not request.get('warmup'):
                # NMS performs an in-place xywh conversion on the NumPy view.
                # Separate copies preserve the untouched output for each gate.
                candidate=graph45.boxes(raw.copy(),confidence=.1)
                high=candidate[candidate[:,4]>=.25]
                if high.shape!=b.shape or not np.allclose(high,b,atol=1e-5,rtol=0):raise RuntimeError('High-score NMS baseline changed')
                selected,audit=associator.update(candidate,float(request['t']))
                extra=dict(candidates=[dict(bbox=r[:4].tolist(),confidence=float(r[4]),class_id=int(r[5])) for r in candidate],
                           associated_boxes=selected,association=audit,high_score_invariant=True)
            print(json.dumps(dict(id=request['id'],boxes=boxes,inference_ms=elapsed,frame_sha256=frame_sha,**extra),allow_nan=False),flush=True)
    finally:image=None;shared.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shared-memory',required=True);p.add_argument('--precision',type=int,choices=[16,32],default=32);p.add_argument('--weights');p.add_argument('--expected-sha256');p.add_argument('--cache-dir');p.add_argument('--body-association',action='store_true');p.add_argument('--frame-rate',type=float,default=25.);a=p.parse_args();main(a.shared_memory,a.precision,a.weights,a.expected_sha256,a.cache_dir,a.body_association,a.frame_rate)
