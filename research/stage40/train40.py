"""Official local transfer learning, with an exact pretrained-parameter check."""
import copy,os,sys,time
from common40 import *
from model37 import load_detector
os.environ['WANDB_DISABLED']='true';os.environ['WANDB_MODE']='disabled';os.environ['COMET_MODE']='DISABLED'

def main():
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            if path==OUT/'targets.json' or path==LABELS/'train3.json' or path.is_relative_to(OUT/'dataset/test') or path.is_relative_to(OUT/'dataset/negative_test'):
                raise RuntimeError('Trainer test-data access forbidden: '+str(path))
    sys.addaudithook(guard)
    import torch,ultralytics
    from ultralytics import YOLO,settings
    for k in ['wandb','mlflow','comet','clearml','neptune','tensorboard','sync','dvc']:
        if k in settings:settings.update({k:False})
    p=verify();prep=load(OUT/'preparation.json');assert sha(OUT/'dataset.yaml')==prep['dataset_yaml_sha256']
    for name,value in prep['label_hashes'].items():assert sha(OUT/name)==value
    assert not (OUT/'train_completion.json').exists();model=load_detector();expected={k:v.detach().cpu().clone() for k,v in model.model.named_parameters()}
    # Locally constructed official module, no original downloaded pickle reload.
    init=OUT/'trusted_local_init.pt';assert not init.exists();torch.save(dict(model=copy.deepcopy(model.model).float(),ema=None,optimizer=None,epoch=-1,best_fitness=None,train_args=dict(task='detect',data=str(OUT/'dataset.yaml'))),init)
    model=YOLO(str(init),task='detect');torch.set_num_threads(2)
    def check_initialization(trainer):
        actual=dict(trainer.model.named_parameters());assert set(actual)==set(expected)
        different=[k for k in expected if not torch.equal(actual[k].detach().cpu(),expected[k])];assert not different,different[:10]
        save(OUT/'initialization_check.json',dict(all_pretrained_parameters_exact=True,parameter_tensors=len(expected),parameter_count=sum(v.numel() for v in expected.values()),upstream_tensor_sha256=load(common37.OUT/'model_metadata.json')['tensor_state_sha256'],trusted_local_checkpoint_sha256=sha(init),torch=torch.__version__,ultralytics=ultralytics.__version__,test_read_guard=True))
    model.add_callback('on_train_start',check_initialization)
    keys=['epochs','imgsz','batch','workers','seed','device','amp','optimizer','lr0','lrf','weight_decay','patience','deterministic','mosaic','mixup','copy_paste','hsv_h','hsv_s','hsv_v','degrees','translate','scale','fliplr','flipud']
    options={k:PARAMS[k] for k in keys};began=time.perf_counter()
    model.train(data=str(OUT/'dataset.yaml'),project=str(OUT),name='training',exist_ok=False,pretrained=True,save=True,save_period=-1,plots=False,verbose=False,cache=False,close_mosaic=0,val=True,**options)
    best=OUT/'training/weights/best.pt';last=OUT/'training/weights/last.pt';assert best.exists() and last.exists() and (OUT/'initialization_check.json').exists()
    save(OUT/'train_completion.json',dict(wall_s=time.perf_counter()-began,best_sha256=sha(best),last_sha256=sha(last),sources_unchanged=verify()['source_hashes']==p['source_hashes'],training_test_read_guard=True,selection='Validation fitness only; fixed20epochs, no test tuning.'))
    print('Training completed; held-out inference pending',flush=True)
if __name__=='__main__':main()
