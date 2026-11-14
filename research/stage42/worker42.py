"""Persistent local GPU worker: one independent native event image per request."""
import base64,contextlib,hashlib,io,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ['YOLO_CONFIG_DIR']=str(ROOT.parent/'tmp/yolo_config37');os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false';os.environ['CUDA_CACHE_PATH']=str(ROOT.parent/'tmp/cuda_cache37')
WEIGHT=ROOT.parent/'output/stage40_record_split_finetune_20261018/training/weights/best.pt'
SHA='ed11fada477ca14ebd4ed0e34a6e73f52cd6112222dbe2941335aa917ee49988'
def main():
    began=time.perf_counter();assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==SHA
    import cv2,numpy as np,torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    with contextlib.redirect_stdout(sys.stderr):
        model=YOLO(str(WEIGHT),task='detect');options=dict(imgsz=640,conf=.25,iou=.45,device=0,verbose=False,save=False,augment=False)
        model.predict(np.zeros((720,1280,3),np.uint8),**options)
    print(json.dumps(dict(ready=True,weight_sha256=SHA,cold_start_s=time.perf_counter()-began)),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):break
        data=base64.b64decode(request['png'],validate=True);image=cv2.imdecode(np.frombuffer(data,np.uint8),1);assert image.shape==(720,1280,3)
        torch.cuda.synchronize();began=time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):result=model.predict(image,**options)[0]
        torch.cuda.synchronize();elapsed=(time.perf_counter()-began)*1000
        boxes=[dict(bbox=b[:4].tolist(),confidence=float(b[4]),class_id=int(b[5])) for b in ([] if result.boxes is None else result.boxes.data.cpu().numpy())]
        print(json.dumps(dict(id=request['id'],boxes=boxes,inference_ms=elapsed),allow_nan=False),flush=True)
if __name__=='__main__':main()
