"""Fresh fixed-backend predictions on original test3/negative5, then score."""
from pathlib import Path
import argparse,hashlib,json,os,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src')]
parser=argparse.ArgumentParser();parser.add_argument('--model-dir',type=Path);parser.add_argument('--no-roi',action='store_true')
parser.add_argument('--baseline-model-dir',type=Path,help='Explicit export.json baseline; default retains the historical original.')
args=parser.parse_args()
OUT=(args.model_dir.resolve() if args.model_dir else WORK/'output/roi_augmented_detector_20261026')/'transfer_v2';OUT.mkdir(exist_ok=False)
PARENT=WORK/'output/stage40_record_split_finetune_20261018'
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(OUT/'local_config'),CUDA_CACHE_PATH=str(ROOT/'.cache/cuda'))
import torch
torch.set_num_threads(2)  # Same CUDA/cuDNN DLL setup order as live_worker.
from forecast_runtime import resolve
from event_drone_forecast import detector_backend as backend_module
from live_roi_refine import refine
export=json.loads((OUT.parent/'export.json').read_text('utf-8'));runtime=resolve()
if args.baseline_model_dir is not None:
    baseline=json.loads((args.baseline_model_dir.resolve()/'export.json').read_text('utf-8'))
    runtime=resolve(detector_weights=baseline['model'],detector_sha256=baseline['sha256'])
inputs=[r for r in json.loads((PARENT/'inputs.json').read_text('utf-8'))['images'] if r['recording'] in [3,5]]
predictions=[];scoring=False
def guard(event,args):
    if not scoring and event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path==PARENT/'targets.json':raise RuntimeError('No targets during inference')
sys.addaudithook(guard)
for name,model_path,expected in [('baseline',runtime['detector_weights'],runtime['detector_sha256']),('adapted',export['model'],export['sha256'])]:
    assert hashlib.sha256(Path(model_path).read_bytes()).hexdigest()==expected
    backend=backend_module.Backend('graph',model_path=Path(model_path))
    for r in inputs:
        image_path=PARENT/r['image'];assert hashlib.sha256(image_path.read_bytes()).hexdigest()==r['image_sha256']
        image=cv2.imdecode(np.fromfile(image_path,np.uint8),cv2.IMREAD_COLOR)
        boxes=backend_module.boxes(backend.run(backend_module.preprocess(image)))
        best=None if not len(boxes) else boxes[int(np.argmax(boxes[:,4]))]
        def encode(row):return None if row is None else dict(bbox=row[:4].tolist(),confidence=float(row[4]))
        predictions.append(dict(id=r['id'],recording=r['recording'],method=name+'_full',body=encode(best)))
        if name=='adapted' and not args.no_roi:
            refined,audit=refine(backend,backend_module,image,best[:4]) if best is not None else (None,dict(reason='no_full_frame_detection'))
            predictions.append(dict(id=r['id'],recording=r['recording'],method='adapted_refined',body=refined or encode(best),refinement=audit))
    del backend
(OUT/'predictions.json').write_text(json.dumps(predictions,indent=2),encoding='utf-8')
scoring=True
truth={r['id']:r for r in json.loads((PARENT/'targets.json').read_text('utf-8'))['targets'] if r['recording'] in [3,5]}
table=[]
for recording in [3,5]:
    for method in ['baseline_full','adapted_full']+([] if args.no_roi else ['adapted_refined']):
        rows=[r for r in predictions if r['recording']==recording and r['method']==method]
        errors=[];ious=[];detected=present=false_positive=absent=success=0
        for r in rows:
            target=truth[r['id']];body=r['body']
            if target['state']=='absent':absent+=1;false_positive+=body is not None;continue
            present+=1
            if body is None:continue
            detected+=1;b=np.asarray(body['bbox']);g=np.asarray(target['bbox'])
            error=np.linalg.norm((b[:2]+b[2:]-g[:2]-g[2:])/2);errors.append(float(error))
            intersection=np.prod(np.maximum(0,np.minimum(b[2:],g[2:])-np.maximum(b[:2],g[:2])))
            union=np.prod(b[2:]-b[:2])+np.prod(g[2:]-g[:2])-intersection
            iou=intersection/max(union,1e-12);ious.append(float(iou));success+=iou>=.3
        table.append(dict(recording=recording,method=method,present=present,detected_present=detected,
            IoU_at_least_point3=int(success),median_IoU=float(np.median(ious)) if ious else None,
            mean_center_error_px=float(np.mean(errors)) if errors else None,
            median_center_error_px=float(np.median(errors)) if errors else None,absent=absent,false_positive_frames=false_positive))
report=dict(table=table,predictions_saved_before_truth_read=True,weights_chosen_before_test=True,
    baseline_model_sha256=runtime['detector_sha256'],adapted_model_sha256=export['sha256'],
    scope='Original record-disjoint test3/negative5 weak labels; historically exposed development, not globally blind. Current640x384 FP32 backends matched.',
    default_promoted=False)
(OUT/'scores.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
