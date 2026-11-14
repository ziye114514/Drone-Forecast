"""Export the single validation-selected local candidate, keep prior weights."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output/roi_augmented_detector_20261026'
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(OUT/'local_config'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
completed=json.loads((OUT/'train_completion.json').read_text('utf-8'))
weight=OUT/'training/weights/best.pt';assert sha(weight)==completed['best_sha256']
target=OUT/'export';target.mkdir(exist_ok=False);local=target/'best.pt';shutil.copyfile(weight,local)
import torch,onnx
from ultralytics import YOLO
torch.set_num_threads(2);model=YOLO(str(local),task='detect');start=time.perf_counter()
exported=Path(model.export(format='onnx',imgsz=[384,640],opset=17,batch=1,device='cpu',
    dynamic=False,simplify=False,nms=False,quantize=32,verbose=False))
assert exported.resolve().is_relative_to(target.resolve())
graph=onnx.load(exported);onnx.checker.check_model(graph)
report=dict(model=str(exported),sha256=sha(exported),source_weight_sha256=sha(weight),
    input_shape=[d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim],
    source_weight_unchanged=sha(weight)==completed['best_sha256'],wall_s=time.perf_counter()-start,
    default_promoted=False,scope='Model export only; no accuracy claim')
(OUT/'export.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
