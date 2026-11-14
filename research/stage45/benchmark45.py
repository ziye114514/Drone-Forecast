import os,time,sys
import cv2,numpy as np
from graph45 import *
os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_CONFIG_DIR']=str(ROOT.parent/'tmp/yolo_config44');os.environ['CUDA_CACHE_PATH']=str(ROOT.parent/'tmp/cuda_cache44')
def main():
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            if path.name=='targets.json' or path.is_relative_to((ROOT.parent/'output/raw_only_project/evaluation_data').resolve()):raise RuntimeError('Truth read forbidden: '+str(path))
    sys.addaudithook(guard)
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    p=verify();torch.set_num_threads(2);model=YOLO(str(common44.OUT/'weights/best.pt'),task='detect');options=dict(imgsz=640,conf=.25,iou=.45,device=0,verbose=False,save=False,augment=False);model.predict(np.zeros((720,1280,3),np.uint8),**options)
    images=[];tensors=[];preprocess_checks=[]
    for r in p['samples']:
        path=common44.INPUT/r['image'];assert sha(path)==r['image_sha256'];im=cv2.imdecode(np.fromfile(path,np.uint8),1);x=model.predictor.preprocess([im]).cpu().numpy();candidate=preprocess(im);preprocess_checks.append(dict(id=r['id'],exact=bool(np.array_equal(candidate,x)),max_abs=float(abs(candidate-x).max())));images.append(im);tensors.append(x)
    backends={};failures={}
    for kind in ['run','binding','graph']:
        try:backends[kind]=Backend(kind)
        except Exception as e:failures[kind]=str(e)
    raw=[];detections=[]
    for i,x in enumerate(tensors):
        with torch.inference_mode():expected=model.predictor.model(torch.from_numpy(x).to('cuda'))
        if isinstance(expected,(tuple,list)):expected=expected[0]
        expected=expected.detach().cpu().numpy()
        for name,b in backends.items():
            actual=b.run(x);delta=abs(actual-expected);raw.append(dict(id=p['samples'][i]['id'],method=name,max_abs=float(delta.max()),score_max_abs=float(delta[:,4:].max())));detections.append(dict(id=p['samples'][i]['id'],method=name,boxes=boxes(actual).tolist()))
        detections.append(dict(id=p['samples'][i]['id'],method='pytorch',boxes=model.predict(images[i],**options)[0].boxes.data.cpu().tolist()))
    stale={}
    for name,b in backends.items():
        first=b.run(tensors[0]).copy();second=b.run(tensors[1]).copy();repeat=b.run(tensors[0]).copy();assert np.array_equal(first,repeat),(name,'repeat not exact');assert not np.array_equal(first,second),(name,'stale input');stale[name]=dict(first_repeat_exact=True,changed_input_changed_output=True)
    def one(name,i,scope):
        if name=='pytorch':
            if scope=='image':model.predict(images[i],**options)
            else:
                with torch.inference_mode():out=model.predictor.model(torch.from_numpy(tensors[i]).to('cuda'))
                if isinstance(out,(tuple,list)):out=out[0]
                b=non_max_suppression(out,conf_thres=.25,iou_thres=.45,nc=1)[0];b[:,:4]=scale_boxes((384,640),b[:,:4],(720,1280));b.cpu().numpy()
        else:boxes(backends[name].run(tensors[i] if scope=='tensor' else preprocess(images[i])))
    names=['pytorch']+list(backends);times={scope:{name:[] for name in names} for scope in ['tensor','image']}
    for scope in times:
        for name in names:
            for i in range(PARAMS['warmup']):one(name,i%len(images),scope)
        for repeat in range(PARAMS['repetitions']):
            for i in range(len(images)):
                shift=(repeat+i)%len(names);ordered=names[shift:]+names[:shift]
                for name in ordered:
                    torch.cuda.synchronize();began=time.perf_counter();one(name,i,scope);torch.cuda.synchronize();times[scope][name].append((time.perf_counter()-began)*1000)
    save(OUT/'benchmark.json',dict(times={scope:{name:dict(n=len(v),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),all_ms=v) for name,v in methods.items()} for scope,methods in times.items()},preprocessing=preprocess_checks,raw_parity=raw,detections=detections,stale_input_checks=stale,failures=failures,backend_metadata={name:dict(provider_options=b.provider_options,initialization_ms=b.initialization_ms) for name,b in backends.items()},sources_unchanged=verify()['source_hashes']==p['source_hashes'],truth_guard=True))
    print('CUDA graph pilot complete',flush=True)
if __name__=='__main__':main()
