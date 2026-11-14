import json,time,sys,os
from pathlib import Path
import cv2,numpy as np
from common37 import ROOT,OUT,PARAMS,load,save,sha,verify
from model37 import load_detector


def boxes(result):
    if result.boxes is None:return []
    return [dict(bbox=r[:4].tolist(),confidence=float(r[4]),class_id=int(r[5])) for r in result.boxes.data.detach().cpu().numpy()]


def main():
    truth_root=(ROOT.parent/'output/raw_only_project/evaluation_data').resolve()
    def reject_truth_access(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            if path.name=='targets.json' or path.is_relative_to(truth_root) or 'freeze_v1_before_inference' in path.parts:
                raise RuntimeError('Evaluation truth access forbidden during detector inference: '+str(path))
    sys.addaudithook(reject_truth_access)
    import torch,ultralytics
    protocol=verify();meta=load(OUT/'model_metadata.json');assert torch.__version__==meta['torch_version'] and ultralytics.__version__==meta['ultralytics_version']
    assert not (OUT/'predictions.jsonl').exists();manifest=load(OUT/'inputs.json');assert sha(OUT/'inputs.json')==load(OUT/'preparation.json')['inputs_sha256']
    model=load_detector();torch.manual_seed(37);torch.set_num_threads(2)
    options=dict(imgsz=PARAMS['imgsz'],conf=PARAMS['confidence'],iou=PARAMS['iou_nms'],device=PARAMS['device'],half=PARAMS['half'],verbose=False,save=False,augment=False)
    model.predict(np.zeros((720,1280,3),np.uint8),**options);torch.cuda.synchronize();timings=[];n=0;first=None;first_images=None
    with (OUT/'predictions.jsonl').open('x',encoding='utf-8') as f:
        for i in range(0,len(manifest['images']),PARAMS['batch_size']):
            rows=manifest['images'][i:i+PARAMS['batch_size']];images=[]
            for r in rows:
                assert sha(OUT/r['image'])==r['image_sha256'];image=cv2.imdecode(np.fromfile(OUT/r['image'],np.uint8),cv2.IMREAD_COLOR);assert image.shape==(r['height'],r['width'],3);images.append(image)
            torch.cuda.synchronize();start=time.perf_counter();rr=model.predict(images,**options);torch.cuda.synchronize();elapsed=time.perf_counter()-start;timings.append(elapsed)
            for r,p in zip(rows,rr):
                record=dict(id=r['id'],case=r['case'],format=r['format'],window_us=r['window_us'],cutoff_us=r['cutoff_us'],boxes=boxes(p),batch_wall_s=elapsed,batch_images=len(rows))
                f.write(json.dumps(record,allow_nan=False)+'\n');n+=1
            if first is None:first=[boxes(r) for r in rr];first_images=images
            if n%40==0:print('Detector images completed: '+str(n),flush=True)
    repeated=[boxes(r) for r in model.predict(first_images,**options)]
    assert len(repeated)==len(first)
    for a,b in zip(first,repeated):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert x['class_id']==y['class_id'] and np.allclose(x['bbox'],y['bbox'],atol=1e-3) and abs(x['confidence']-y['confidence'])<1e-5
    save(OUT/'inference.json',dict(images=n,gpu=torch.cuda.get_device_name(0),batch_size=PARAMS['batch_size'],sum_batch_wall_s=sum(timings),
        median_batch_ms=float(np.median(timings)*1000),p95_batch_ms=float(np.percentile(timings,95)*1000),first_batch_repeat=True,
        inputs_sha256=sha(OUT/'inputs.json'),predictions_sha256=sha(OUT/'predictions.jsonl'),sources_unchanged=verify()['source_hashes']==protocol['source_hashes'],
        labels_read=False,ground_truth_file_access_guard=True,scope='Snapshot migration only; batching/warm timings are not complete RAW real-time performance. No future predictions produced.'))
    print('All neural detector predictions saved without label reads',flush=True)


if __name__=='__main__':main()
