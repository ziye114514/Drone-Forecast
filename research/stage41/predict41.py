import os,sys,time
import cv2,numpy as np
from common41 import *
from model37 import load_detector

def main():
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            if path.is_relative_to(common40.LABELS.resolve()) or path.name=='targets.json' or 'labels' in path.parts and path.is_relative_to(PARENT):raise RuntimeError('Inference truth read forbidden: '+str(path))
    sys.addaudithook(guard)
    import torch
    from ultralytics import YOLO
    p=verify();meta=load(PARENT/'train_completion.json');best=PARENT/'training/weights/best.pt';assert sha(best)==meta['best_sha256'];assert sha(OUT/'inputs.json')==load(OUT/'preparation.json')['input_sha256'];rows=load(OUT/'inputs.json')['images'];assert not (OUT/'detections.jsonl').exists();timings={};torch.set_num_threads(2)
    with (OUT/'detections.jsonl').open('x',encoding='utf-8') as f:
        for method in ['pretrained','finetuned']:
            model=load_detector() if method=='pretrained' else YOLO(str(best),task='detect');tt=[];options=dict(imgsz=640,conf=.25,iou=.45,device=0,verbose=False,save=False,augment=False)
            model.predict(np.zeros((720,1280,3),np.uint8),**options)
            for i in range(0,len(rows),4):
                batch=rows[i:i+4];images=[]
                for r in batch:assert sha(OUT/r['image'])==r['image_sha256'];images.append(cv2.imdecode(np.fromfile(OUT/r['image'],np.uint8),1))
                torch.cuda.synchronize();began=time.perf_counter();results=model.predict(images,**options);torch.cuda.synchronize();tt.append(time.perf_counter()-began)
                for r,result in zip(batch,results):
                    boxes=[dict(bbox=b[:4].tolist(),confidence=float(b[4]),class_id=int(b[5])) for b in ([] if result.boxes is None else result.boxes.data.cpu().numpy())]
                    f.write(json.dumps(dict(id=r['id'],case=r['case'],t=r['t'],method=method,boxes=boxes),allow_nan=False)+'\n')
            timings[method]=dict(frames=len(rows),batch_wall_s=sum(tt),median_batch_ms=float(np.median(tt)*1000),p95_batch_ms=float(np.percentile(tt,95)*1000));print(method,'continuous detections saved',len(rows),flush=True)
    save(OUT/'inference.json',dict(timings=timings,detector_weight_sha256=meta['best_sha256'],predictions_sha256=sha(OUT/'detections.jsonl'),ground_truth_guard=True,sources_unchanged=verify()['source_hashes']==p['source_hashes']))
if __name__=='__main__':main()
