"""One official ORT conversion, fixed19-image numerical/timing probe."""
from pathlib import Path
import hashlib,json,os,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'src')]
OUT=WORK/'output/mixed_fp16_detector_v3_20261026';OUT.mkdir(exist_ok=False)
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(OUT/'local_config'),CUDA_CACHE_PATH=str(ROOT/'.cache/cuda'))
PARENT=WORK/'output/equal_weight_detector_20261026';export=json.loads((PARENT/'export.json').read_text('utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=Path(export['model']);assert sha(source)==export['sha256']
inputs=json.loads((WORK/'output/stage40_record_split_finetune_20261018/inputs.json').read_text('utf-8'))['images']
chosen=[r for r in inputs if r['recording']==3][:12]+[r for r in inputs if r['recording']==5]
assert len(chosen)==19
design=dict(source_sha256=export['sha256'],conversion='Installed onnxruntime.transformers.float16.convert_float_to_float16 keep_io_types=True; default blocked operators/constant bounds.',
    reference='https://onnxruntime.ai/docs/performance/model-optimizations/float16.html',chosen_ids=[r['id'] for r in chosen],
    gate='Same primary detection presence and NMS box counts; maximum selected center shift<=1px and confidence difference<=.02. Smaller graph; measured forward median lower.',
    threshold_sweep=False,training_or_labels_read=False,default_promoted=False)
(OUT/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
import torch,onnx
torch.set_num_threads(2)
from onnxruntime.transformers.float16 import convert_float_to_float16
model=convert_float_to_float16(onnx.load(source),keep_io_types=True)
from onnxruntime.transformers.onnx_model import OnnxModel
OnnxModel(model).topological_sort()  # Official helper orders newly inserted IO casts.
metadata=model.metadata_props.add();metadata.key='edf.internal_precision';metadata.value='mixed_fp16_with_fp32_IO'
onnx.checker.check_model(model);assert model.graph.input[0].type.tensor_type.elem_type==1 and model.graph.output[0].type.tensor_type.elem_type==1
target=OUT/'mixed_fp16.onnx';onnx.save(model,target)
from event_drone_forecast import detector_backend as backend_module
engines=dict(fp32=backend_module.Backend('graph',model_path=source),mixed=backend_module.Backend('graph',model_path=target))
costs={name:[] for name in engines};rows=[];max_shift=max_confidence=0.;same=True
for index,r in enumerate(chosen):
    image_path=WORK/'output/stage40_record_split_finetune_20261018'/r['image']
    assert sha(image_path)==r['image_sha256']
    image=cv2.imdecode(np.fromfile(image_path,np.uint8),cv2.IMREAD_COLOR);x=backend_module.preprocess(image)
    boxes={};raw={}
    for name in (['fp32','mixed'] if index%2==0 else ['mixed','fp32']):
        start=time.perf_counter();value=engines[name].run(x);costs[name].append((time.perf_counter()-start)*1000)
        assert value.shape==(1,5,5040) and np.isfinite(value).all();raw[name]=value.copy()
        boxes[name]=backend_module.boxes(value.copy())
    a,b=boxes['fp32'],boxes['mixed'];equal=len(a)==len(b);same&=equal
    shift=confidence=None
    if len(a) and len(b):
        a=a[np.argmax(a[:,4])];b=b[np.argmax(b[:,4])]
        shift=float(np.linalg.norm((a[:2]+a[2:4]-b[:2]-b[2:4])/2));confidence=float(abs(a[4]-b[4]))
        max_shift=max(max_shift,shift);max_confidence=max(max_confidence,confidence)
    rows.append(dict(id=r['id'],same_box_count=equal,center_shift_px=shift,confidence_difference=confidence,
        raw_max_absolute_difference=float(np.max(abs(raw['fp32']-raw['mixed'])))))
performance={name:dict(median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95))) for name,values in costs.items()}
passed=bool(same and max_shift<=1. and max_confidence<=.02 and target.stat().st_size<source.stat().st_size and performance['mixed']['median_ms']<performance['fp32']['median_ms'])
report=dict(design=design,model=str(target),sha256=sha(target),source_bytes=source.stat().st_size,mixed_bytes=target.stat().st_size,
    internal_half_initializers=sum(t.data_type==10 for t in model.graph.initializer),input_output_precision=32,
    same_box_counts=same,maximum_center_shift_px=max_shift,maximum_confidence_difference=max_confidence,
    forward=performance,provisional_gate_passed=passed,rows=rows,
    trajectory_accuracy_or_end_to_end_speedup_proven=False,default_promoted=False,
    source_unchanged=sha(source)==export['sha256'],single_interleaved_probe_not_hardware_guarantee=True)
(OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['design','rows']}),flush=True)
