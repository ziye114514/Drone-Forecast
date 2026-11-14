"""One official Ultralytics fine-tune; prohibit test data during training."""
from pathlib import Path
import hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
OUT=WORK/'output/roi_augmented_detector_20261026';PARENT=WORK/'output/stage40_record_split_finetune_20261018'
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(OUT/'local_config'),
    CUDA_CACHE_PATH=str(ROOT/'.cache/cuda'),WANDB_DISABLED='true',WANDB_MODE='disabled',COMET_MODE='DISABLED')
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(PARENT/'dataset/test') or path.is_relative_to(PARENT/'dataset/negative_test') or path.name in ['dense_labels_train2_40_46','labels.json']:
            raise RuntimeError('Test or dense forecast labels forbidden during fine-tuning: '+str(path))
sys.addaudithook(guard)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
design=json.loads((OUT/'protocol.json').read_text('utf-8'))
assert sha(Path(design['parent_weight']))==design['parent_weight_sha256']
assert not (OUT/'training').exists()
import torch
from ultralytics import YOLO,settings
torch.set_num_threads(2)
for name in ['wandb','mlflow','comet','clearml','neptune','tensorboard','sync','dvc']:
    if name in settings:settings.update({name:False})
model=YOLO(design['parent_weight'],task='detect')
keys=['epochs','imgsz','rect','batch','workers','seed','device','amp','optimizer','lr0','lrf','weight_decay','patience','deterministic','mosaic','mixup','copy_paste','hsv_h','hsv_s','hsv_v','degrees','translate','scale','fliplr','flipud']
begin=time.perf_counter()
def record_epoch(trainer):
    data=dict(epoch=int(trainer.epoch)+1,total_epochs=int(trainer.epochs),elapsed_s=time.perf_counter()-begin)
    (OUT/'progress.json').write_text(json.dumps(data),encoding='utf-8')
model.add_callback('on_train_epoch_end',record_epoch)
model.train(data=str(OUT/'dataset.yaml'),project=str(OUT),name='training',exist_ok=False,
    pretrained=True,freeze=design['freeze_gradient_first_layers'],save=True,save_period=-1,
    plots=False,verbose=False,cache=False,close_mosaic=0,val=True,**{k:design[k] for k in keys})
best=OUT/'training/weights/best.pt';last=OUT/'training/weights/last.pt'
assert best.exists() and last.exists()
result=dict(wall_s=time.perf_counter()-begin,best_sha256=sha(best),last_sha256=sha(last),
    parent_unchanged=sha(Path(design['parent_weight']))==design['parent_weight_sha256'],
    selection=design['selection'],test_data_blocked=True,default_promoted=False)
(OUT/'train_completion.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
