"""Fixed small official-Ultralytics fine-tune; scoring data are blocked."""
from pathlib import Path
import hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
OUT=WORK/'output/fred_domain_adaptation_20261029'
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(OUT/'local_config'),
    CUDA_CACHE_PATH=str(ROOT/'.cache/cuda'),WANDB_DISABLED='true',WANDB_MODE='disabled',COMET_MODE='DISABLED')


def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(WORK/'output/fred_transfer_20261029') or any(x.startswith('dense_labels') for x in path.parts):
            raise RuntimeError('Forecast scoring data blocked during adaptation: '+str(path))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    design=json.loads((OUT/'protocol.json').read_text('utf-8'));source=Path(design['parent_weight'])
    assert sha(source)==design['parent_weight_sha256'] and not (OUT/'training').exists()
    sys.addaudithook(guard)
    import torch
    from ultralytics import YOLO,settings
    torch.set_num_threads(2)
    for name in ['wandb','mlflow','comet','clearml','neptune','tensorboard','sync','dvc']:
        if name in settings:settings.update({name:False})
    model=YOLO(str(source),task='detect');begin=time.perf_counter()
    def progress(trainer):
        (OUT/'progress.json').write_text(json.dumps(dict(epoch=int(trainer.epoch)+1,total_epochs=design['epochs'],
            elapsed_s=time.perf_counter()-begin)),'utf-8')
    model.add_callback('on_train_epoch_end',progress)
    keys=['epochs','imgsz','rect','batch','workers','seed','device','amp','optimizer','lr0','lrf','weight_decay','patience','deterministic',
        'mosaic','mixup','copy_paste','hsv_h','hsv_s','hsv_v','degrees','translate','scale','fliplr','flipud']
    model.train(data=str(OUT/'dataset.yaml'),project=str(OUT),name='training',exist_ok=False,
        pretrained=True,freeze=design['freeze'],save=True,save_period=-1,plots=False,verbose=False,
        cache=False,close_mosaic=0,val=True,**{k:design[k] for k in keys})
    best=OUT/'training/weights/best.pt';last=OUT/'training/weights/last.pt'
    assert best.is_file() and last.is_file() and sha(source)==design['parent_weight_sha256']
    result=dict(wall_s=time.perf_counter()-begin,best_sha256=sha(best),last_sha256=sha(last),
        parent_unchanged=True,scoring_data_blocked=True,selection=design['selection'],default_model_unchanged=True)
    (OUT/'train_completion.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
