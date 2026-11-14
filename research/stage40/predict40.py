"""Truth-guarded paired original/fine-tuned detector inference."""
import os,sys,time
import cv2,numpy as np
from common40 import *
from model37 import load_detector

def main():
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            if path==OUT/'targets.json' or path.is_relative_to(LABELS.resolve()) or 'labels' in path.parts and path.is_relative_to(OUT):raise RuntimeError('Inference truth forbidden: '+str(path))
    sys.addaudithook(guard)
    import torch
    from ultralytics import YOLO
    p=verify();prep=load(OUT/'preparation.json');assert sha(OUT/'inputs.json')==prep['input_sha256'];meta=load(OUT/'train_completion.json');best=OUT/'training/weights/best.pt';assert sha(best)==meta['best_sha256']
    rows=[r for r in load(OUT/'inputs.json')['images'] if r['split']!='train'];torch.set_num_threads(2);torch.manual_seed(40);timings={};assert not (OUT/'predictions.jsonl').exists()
    with (OUT/'predictions.jsonl').open('x',encoding='utf-8') as f:
        for method in ['pretrained','finetuned']:
            model=load_detector() if method=='pretrained' else YOLO(str(best),task='detect');tt=[]
            options=dict(imgsz=PARAMS['imgsz'],conf=PARAMS['confidence'],iou=PARAMS['iou_nms'],device=0,verbose=False,save=False,augment=False)
            model.predict(np.zeros((720,1280,3),np.uint8),**options)
            for i in range(0,len(rows),PARAMS['inference_batch']):
                batch=rows[i:i+PARAMS['inference_batch']];images=[]
                for r in batch:assert sha(OUT/r['image'])==r['image_sha256'];images.append(cv2.imdecode(np.fromfile(OUT/r['image'],np.uint8),cv2.IMREAD_COLOR))
                torch.cuda.synchronize();began=time.perf_counter();pred=model.predict(images,**options);torch.cuda.synchronize();tt.append(time.perf_counter()-began)
                for r,result in zip(batch,pred):
                    boxes=[dict(bbox=b[:4].tolist(),confidence=float(b[4]),class_id=int(b[5])) for b in ([] if result.boxes is None else result.boxes.data.cpu().numpy())]
                    f.write(json.dumps(dict(id=r['id'],method=method,recording=r['recording'],split=r['split'],cutoff_us=r['cutoff_us'],boxes=boxes),allow_nan=False)+'\n')
            timings[method]=dict(images=len(rows),batch_wall_s=sum(tt),median_batch_ms=float(np.median(tt)*1000),p95_batch_ms=float(np.percentile(tt,95)*1000));print('Predictions saved',method,len(rows),flush=True)
    save(OUT/'inference.json',dict(timings=timings,predictions_sha256=sha(OUT/'predictions.jsonl'),truth_guard=True,labels_read=False,sources_unchanged=verify()['source_hashes']==p['source_hashes']))
if __name__=='__main__':main()
