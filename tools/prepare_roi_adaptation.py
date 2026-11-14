"""Local crop augmentation of existing40 training/validation images only."""
from pathlib import Path
import hashlib,json,shutil,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_roi_refine import crop_transform,DESIGN
PARENT=WORK/'output/stage40_record_split_finetune_20261018'
OUT=WORK/'output/roi_augmented_detector_20261026';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
weight=PARENT/'training/weights/best.pt'
assert sha(weight)==json.loads((PARENT/'train_completion.json').read_text('utf-8'))['best_sha256']
rows=[]
for split in ['train','val']:
    target=OUT/'dataset'/split
    for kind in ['images','labels']:(target/kind).mkdir(parents=True)
    for image in sorted((PARENT/'dataset'/split/'images').glob('*.png')):
        label=PARENT/'dataset'/split/'labels'/(image.stem+'.txt')
        shutil.copyfile(image,target/'images'/image.name);shutil.copyfile(label,target/'labels'/label.name)
        text=label.read_text('utf-8').strip();item=dict(split=split,source_image=str(image),source_sha256=sha(image),label_sha256=sha(label),augmented=False)
        if text:
            values=text.split();assert len(values)==5 and values[0]=='0'
            cx,cy,bw,bh=np.asarray(values[1:],float)*[1280,720,1280,720]
            box=np.array([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2]);transform=crop_transform(box)
            if transform is not None:
                origin,scale,matrix=transform;frame=cv2.imdecode(np.fromfile(image,np.uint8),cv2.IMREAD_COLOR)
                assert frame.shape==(720,1280,3)
                crop=cv2.warpAffine(frame,matrix,(1280,720),flags=cv2.INTER_LINEAR)
                b=(box-np.tile(origin,2))*scale;c=(b[:2]+b[2:])/2;size=b[2:]-b[:2]
                norm=np.r_[c,size]/[1280,720,1280,720]
                assert np.all(norm>0) and np.all(norm<=1)
                name=image.stem+'_roi';dest=target/'images'/(name+'.png')
                cv2.imencode('.png',crop)[1].tofile(dest)
                (target/'labels'/(name+'.txt')).write_text('0 '+' '.join(f'{v:.9f}' for v in norm)+'\n',encoding='utf-8')
                item.update(augmented=True,augmentation_sha256=sha(dest),origin=origin.tolist(),scale=scale)
        rows.append(item)
(OUT/'dataset.yaml').write_text('path: '+(OUT/'dataset').as_posix()+'\ntrain: train/images\nval: val/images\nnames:\n  0: drone\n',encoding='utf-8')
design=dict(parent_weight=str(weight),parent_weight_sha256=sha(weight),
    split=dict(train_recordings=[1,2,4],validation_recordings=[6],test_recordings=[3]),
    crop_augmentation=DESIGN,ground_truth_crop_used_only_for_training_or_validation=True,
    inference_requires_current_detected_box=True,epochs=10,freeze_gradient_first_layers=10,
    imgsz=640,rect=True,batch=8,workers=0,seed=2610,device=0,amp=False,
    optimizer='AdamW',lr0=.0001,lrf=.1,weight_decay=.0005,patience=1000,
    deterministic=True,mosaic=0.,mixup=0.,copy_paste=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,
    degrees=0.,translate=0.,scale=0.,fliplr=.5,flipud=0.,
    selection='One fixed10epoch run, validation fitness of full+ROI images only. No test or forecast-label tuning.',
    parameters_swept=False,default_promoted=False,related_project='https://github.com/obss/sahi',
    method='Established crop augmentation and official Ultralytics fine-tuning, not a SAHI reproduction or novel network.',
    scope='Original weak-box record split, previously exposed development data; not a globally blind benchmark.')
for name in ['tools/prepare_roi_adaptation.py','tools/train_roi_adaptation.py','live_roi_refine.py']:
    dest=OUT/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
manifest=dict(design=design,images={split:dict(original=sum(r['split']==split for r in rows),
    crops=sum(r['split']==split and r['augmented'] for r in rows)) for split in ['train','val']},
    dataset_yaml_sha256=sha(OUT/'dataset.yaml'),rows=rows)
(OUT/'preparation.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(OUT/'protocol.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
print(json.dumps(dict(images=manifest['images'],parent_weight_sha256=sha(weight),test_images_read=False)),flush=True)
