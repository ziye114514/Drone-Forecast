"""One matched-control pair, fixed five epochs per arm; no forecast labels."""
from pathlib import Path
import argparse,copy,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['control','consistency'],required=True)
a=parser.parse_args();OUT=WORK/'output/flip_consistency_20261026';ARM=OUT/a.arm
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(ARM/'local_config'),
    CUDA_CACHE_PATH=str(ROOT/'.cache/cuda'),WANDB_DISABLED='true',WANDB_MODE='disabled',COMET_MODE='DISABLED')

def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve()
        if any(part in ['test','negative_test','dense_labels_side_110_1195','dense_labels_train2_40_46'] for part in p.parts) or p.name in ['labels.json','scores.json']:
            raise RuntimeError('Test/forecast labels forbidden during training: '+str(p))
sys.addaudithook(guard)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
design=json.loads((OUT/'design.json').read_text('utf-8'))
parent=Path(design['parent_weight']);assert sha(parent)==design['parent_sha256']
assert not (ARM/'training').exists();ARM.mkdir(exist_ok=True)
import torch,onnx
from ultralytics import YOLO,settings
from ultralytics.nn.tasks import DetectionModel
from consistency_detector import ConsistencyTrainer
torch.set_num_threads(2)
for name in ['wandb','mlflow','comet','clearml','neptune','tensorboard','sync','dvc']:
    if name in settings:settings.update({name:False})
class Trainer(ConsistencyTrainer):consistency_weight=0. if a.arm=='control' else design['consistency_weight']
model=YOLO(str(parent),task='detect');start=time.perf_counter();audits=[];batch_audits=[]
def batch_progress(trainer):
    batch_audits.append(dict(getattr(trainer.model,'consistency_audit',{})))
def progress(trainer):
    sample=dict(epoch=trainer.epoch+1,elapsed_s=time.perf_counter()-start,batches=len(batch_audits),
        matched_pairs=sum(r.get('pairs',0) for r in batch_audits),targets=sum(r.get('targets',0) for r in batch_audits),
        auxiliary_mean=sum(r.get('loss',0) for r in batch_audits)/max(len(batch_audits),1))
    audits.append(sample);(ARM/'progress.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    batch_audits.clear()
model.add_callback('on_train_batch_end',batch_progress)
model.add_callback('on_train_epoch_end',progress)
model.train(trainer=Trainer,data=design['dataset'],project=str(ARM),name='training',exist_ok=False,
    pretrained=True,freeze=10,epochs=5,imgsz=640,rect=True,batch=4,workers=0,seed=261026,device=0,amp=False,
    optimizer='AdamW',lr0=.00005,lrf=.1,weight_decay=.0005,patience=1000,deterministic=True,
    mosaic=0.,mixup=0.,copy_paste=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,degrees=0.,translate=0.,scale=0.,fliplr=0.,flipud=0.,
    save=True,save_period=-1,plots=False,verbose=False,cache=False,close_mosaic=0,val=True)
best=ARM/'training/weights/best.pt';last=ARM/'training/weights/last.pt'
assert best.exists() and last.exists()
# Discard the training-only subclass/criterion. Deployment has no extra branch.
trained=YOLO(str(best),task='detect').model.float().cpu()
plain=DetectionModel(trained.yaml,nc=1,ch=3,verbose=False);plain.load_state_dict(trained.state_dict(),strict=True)
plain.names=trained.names;plain.args=dict(task='detect')
weight=ARM/'deploy.pt'
torch.save(dict(model=copy.deepcopy(plain).float(),ema=None,optimizer=None,epoch=-1,best_fitness=None,train_args=dict(task='detect')),weight)
exported=Path(YOLO(str(weight),task='detect').export(format='onnx',imgsz=[384,640],opset=17,batch=1,device='cpu',dynamic=False,
    simplify=False,nms=False,quantize=32,verbose=False))
onnx.checker.check_model(onnx.load(exported))
report=dict(arm=a.arm,model=str(exported),sha256=sha(exported),source_weight_sha256=sha(weight),
    training_best_sha256=sha(best),training_last_sha256=sha(last),wall_s=time.perf_counter()-start,
    parent_unchanged=sha(parent)==design['parent_sha256'],test_and_forecast_labels_blocked=True,
    selection='Validation fitness only; five epochs fixed before training.',deployment_extra_branches=0)
(ARM/'export.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
