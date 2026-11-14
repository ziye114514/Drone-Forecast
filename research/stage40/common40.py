"""A fixed record-disjoint, weak-box domain adaptation experiment."""
from pathlib import Path
import datetime,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage37'))
import common37
OUT=ROOT.parent/'output/stage40_record_split_finetune_20261018'
LABELS=ROOT.parent/'output/raw_only_project/evaluation_data'
SPLIT={'train':[1,2,4],'val':[6],'test':[3]}
PARAMS=dict(window_us=20000,warmup_us=10000,activity_threshold_us=10000,cutoff='floor(target_t/20ms)*20ms',palette='SDK default native ON/OFF, no rebin/G',
    epochs=20,imgsz=640,batch=8,workers=0,seed=40,device=0,amp=False,optimizer='AdamW',lr0=.001,lrf=.1,weight_decay=.0005,
    patience=1000,deterministic=True,mosaic=0.,mixup=0.,copy_paste=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,degrees=0.,translate=.1,scale=.5,fliplr=.5,flipud=0.,
    confidence=.25,iou_nms=.45,inference_batch=4,
    primary='Best validation-fitness checkpoint of one fixed20-epoch run. Original certified pretrained detector and tuned detector on exactly same images. Highest confidence box is primary, no GT-based box selection.',
    scope='Existing human approximate visible whole-UAV boxes, not COM/pose/RPM. Test recording excluded from gradient/validation selection, but seen by previous development stages, NOT globally blind. No test-driven hyperparameter tuning.')

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sources():
    result=common37.sources()
    for n in ['common40.py','prepare40.py','train40.py','predict40.py','score40.py']:result[str((Path(__file__).parent/n).relative_to(ROOT))]=sha(Path(__file__).parent/n)
    return result
def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];common37.verify();return p
def freeze():
    assert not (OUT/'protocol.json').exists()
    files={f'train{i}.json':sha(LABELS/f'train{i}.json') for v in SPLIT.values() for i in v}
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parameters=PARAMS,split=SPLIT,label_hashes=files,
        upstream='takara-ai/DroneStalker fixed93e846eef8abc974da1e0c5cf71331729648f143, reviewed EMA exported to official YAML + tensor-only state in37. Ultralytics installed official training API; no new network design claim.',
        references=['https://github.com/takara-ai/DroneStalker','https://docs.ultralytics.com/modes/train/'],
        design='Freeze before data preparation/training. Every existing label on each specified recording, no label interpolation or positive/zero-event exclusion. Keep3 entirely outside training/validation; seven known-empty5 timestamps as additional test negatives. Train/val truth available to trainer; test inference truth forbidden. Model input full native frame, never GT crop.',
        promotion='First verify actual pretrained initialization, current coverage/IoU/false positives and record transfer; do not equate detector improvement to future prediction or paper reproduction. No old/default/protected files altered.'))
    print('Stage40 frozen',flush=True)
if __name__=='__main__':freeze()
