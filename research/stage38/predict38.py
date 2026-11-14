import json,time,sys,os
from pathlib import Path
import cv2,numpy as np
from common38 import ROOT,OUT,OLD,PARAMS,load,save,sha,verify,source_box
from model37 import load_detector


def main():
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.name in ['targets.json','control_targets.json'] or p.is_relative_to((ROOT.parent/'output/raw_only_project/evaluation_data').resolve()) or 'freeze_v1_before_inference' in p.parts:
                raise RuntimeError('Detector evaluation truth read forbidden: '+str(p))
    sys.addaudithook(guard)
    import torch,ultralytics
    protocol=verify();meta=load(OLD/'model_metadata.json');assert meta['torch_version']==torch.__version__ and meta['ultralytics_version']==ultralytics.__version__
    assert not (OUT/'predictions.jsonl').exists();manifest=load(OUT/'inputs.json')['images'];assert sha(OUT/'inputs.json')==load(OUT/'preparation.json')['input_hash'];model=load_detector();torch.set_num_threads(2);torch.manual_seed(38)
    options=dict(imgsz=PARAMS['imgsz'],conf=PARAMS['confidence'],iou=PARAMS['iou_nms'],device=PARAMS['device'],verbose=False,save=False,augment=False)
    model.predict(np.zeros((720,1280,3),np.uint8),**options);assert next(model.model.parameters()).dtype==torch.float32;timings=[];n=0
    with (OUT/'predictions.jsonl').open('x',encoding='utf-8') as f:
        for i in range(0,len(manifest),PARAMS['batch_size']):
            rows=manifest[i:i+PARAMS['batch_size']];images=[]
            for r in rows:
                assert sha(OUT/r['image'])==r['image_sha256'];images.append(cv2.imdecode(np.fromfile(OUT/r['image'],np.uint8),cv2.IMREAD_COLOR))
            torch.cuda.synchronize();began=time.perf_counter();pred=model.predict(images,**options);torch.cuda.synchronize();elapsed=time.perf_counter()-began;timings.append(elapsed)
            for r,p in zip(rows,pred):
                result=[]
                for b in ([] if p.boxes is None else p.boxes.data.cpu().numpy()):
                    mapped=source_box(b[:4].tolist(),r['viewport'],r['divisor'],r['width'],r['height'])
                    if mapped is not None:result.append(dict(bbox=mapped,raw_view_bbox=b[:4].tolist(),confidence=float(b[4]),class_id=int(b[5])))
                f.write(json.dumps(dict(id=r['id'],case=r['case'],noise=r['noise'],divisor=r['divisor'],window_us=r['window_us'],cutoff_us=r['cutoff_us'],boxes=result),allow_nan=False)+'\n');n+=1
            if n%80==0:print('Noise/scale images completed: '+str(n),flush=True)
    save(OUT/'inference.json',dict(images=n,predictions_sha256=sha(OUT/'predictions.jsonl'),input_sha256=sha(OUT/'inputs.json'),sum_batch_wall_s=sum(timings),
        median_batch_ms=float(np.median(timings)*1000),p95_batch_ms=float(np.percentile(timings,95)*1000),device=torch.cuda.get_device_name(0),ground_truth_access_guard=True,labels_read=False,sources_unchanged=verify()['source_hashes']==protocol['source_hashes']))
    print('All noise/scale predictions saved',flush=True)


if __name__=='__main__':main()
