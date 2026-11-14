"""One fixed weight average, related to WiSE-FT/model soups, no alpha sweep."""
from pathlib import Path
import copy,hashlib,json,os,time
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
OUT=WORK/'output/equal_weight_detector_20261026';OUT.mkdir(exist_ok=False)
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(OUT/'local_config'))
old=WORK/'output/stage40_record_split_finetune_20261018/training/weights/best.pt'
fine=WORK/'output/roi_augmented_detector_20261026/training/weights/best.pt'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old)==json.loads((old.parents[2]/'train_completion.json').read_text('utf-8'))['best_sha256']
assert sha(fine)==json.loads((fine.parents[2]/'train_completion.json').read_text('utf-8'))['best_sha256']
design=dict(alpha=.5,alpha_sweep=False,kind='Equal-weight interpolation of original and crop-adapted detector',
    primary_reference='https://github.com/mlfoundations/wise-ft/blob/master/src/wise_ft.py',
    related_paper='https://proceedings.mlr.press/v162/wortsman22a.html',
    implementation='PyTorch torch.lerp on floating state tensors; original integer counters retained.',
    parent0_sha256=sha(old),parent1_sha256=sha(fine),extra_inference_models=0,
    goal='Avoid observed latency cost of dual inference, not claim paper reproduction or novelty.',
    selection='Single alpha .5 fixed before merged-model evaluation, exploratory after dual model results; no label-based weight choice.',
    default_promoted=False)
(OUT/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
import torch,onnx
from ultralytics import YOLO
torch.set_num_threads(2);a=YOLO(str(old),task='detect');b=YOLO(str(fine),task='detect')
s0=a.model.float().state_dict();s1=b.model.float().state_dict();assert s0.keys()==s1.keys()
merged={};floating=integral=changed=0
for key in s0:
    assert s0[key].shape==s1[key].shape and s0[key].dtype==s1[key].dtype
    if s0[key].is_floating_point():
        assert torch.isfinite(s0[key]).all() and torch.isfinite(s1[key]).all()
        assert torch.equal(torch.lerp(s0[key],s1[key],0.),s0[key])
        assert torch.equal(torch.lerp(s0[key],s1[key],1.),s1[key])
        merged[key]=torch.lerp(s0[key],s1[key],.5);floating+=1;changed+=not torch.equal(s0[key],merged[key])
    else:merged[key]=s0[key].clone();integral+=1
a.model.load_state_dict(merged,strict=True)
weight=OUT/'merged.pt'
torch.save(dict(model=copy.deepcopy(a.model).float(),ema=None,optimizer=None,epoch=-1,best_fitness=None,train_args=dict(task='detect')),weight)
model=YOLO(str(weight),task='detect');start=time.perf_counter()
exported=Path(model.export(format='onnx',imgsz=[384,640],opset=17,batch=1,device='cpu',dynamic=False,
    simplify=False,nms=False,quantize=32,verbose=False))
graph=onnx.load(exported);onnx.checker.check_model(graph)
report=dict(model=str(exported),sha256=sha(exported),source_weight_sha256=sha(weight),kind=design['kind'],
    input_shape=[d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim],
    floating_tensors=floating,original_integral_tensors=integral,changed_float_tensors=changed,
    parent0_unchanged=sha(old)==design['parent0_sha256'],parent1_unchanged=sha(fine)==design['parent1_sha256'],
    wall_export_s=time.perf_counter()-start,default_promoted=False,accuracy_claimed=False)
(OUT/'export.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(OUT/'build_equal_weight_detector.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(report),flush=True)
