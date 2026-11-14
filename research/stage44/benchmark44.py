"""Independent per-frame benchmark; explicit EP and exact-input tensor checks."""
import os,time
import cv2,numpy as np
from common44 import *
os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_CONFIG_DIR']=str(ROOT.parent/'tmp/yolo_config44');os.environ['CUDA_CACHE_PATH']=str(ROOT.parent/'tmp/cuda_cache44')
def main():
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            if path.name=='targets.json' or path.is_relative_to((ROOT.parent/'output/raw_only_project/evaluation_data').resolve()):raise RuntimeError('Benchmark truth access forbidden: '+str(path))
    sys.addaudithook(guard)
    import torch,onnxruntime as ort
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    p=verify();meta=load(OUT/'export.json');path=OUT/meta['model'];assert sha(path)==meta['sha256'];torch.set_num_threads(2)
    model=YOLO(str(OUT/'weights/best.pt'),task='detect');images=[]
    for r in p['samples']:assert sha(INPUT/r['image'])==r['image_sha256'];images.append(cv2.imdecode(np.fromfile(INPUT/r['image'],np.uint8),1))
    options=dict(imgsz=640,conf=.25,iou=.45,device=0,verbose=False,save=False,augment=False);model.predict(np.zeros((720,1280,3),np.uint8),**options)
    tensors=[]
    for im in images:
        x=model.predictor.preprocess([im]);assert tuple(x.shape)==(1,3,384,640) and x.dtype==torch.float32;tensors.append(x.cpu().numpy())
    sessions={};failures={};profile_paths={}
    for name,provider in [('ort_cuda','CUDAExecutionProvider'),('ort_cpu','CPUExecutionProvider')]:
        try:
            so=ort.SessionOptions();so.intra_op_num_threads=2;so.inter_op_num_threads=1;so.enable_profiling=True;so.profile_file_prefix=str(OUT/name);so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session=ort.InferenceSession(str(path),sess_options=so,providers=[provider]);assert provider in session.get_providers(),session.get_providers();session.disable_fallback();sessions[name]=session
        except Exception as e:failures[name]=str(e)
    output_names={name:[r.name for r in session.get_outputs()] for name,session in sessions.items()};raw=[];detections=[]
    def torch_raw(x):
        y=model.predictor.model(torch.from_numpy(x).to('cuda'));return y[0] if isinstance(y,(tuple,list)) else y
    for i,x in enumerate(tensors):
        expected=torch_raw(x).detach().cpu().numpy()
        for name,session in sessions.items():
            actual=session.run(output_names[name],{session.get_inputs()[0].name:x})[0];assert actual.shape==expected.shape
            delta=abs(actual-expected);raw.append(dict(id=p['samples'][i]['id'],method=name,shape=list(actual.shape),max_abs=float(delta.max()),mean_abs=float(delta.mean()),coordinate_max_abs=float(delta[:,:4].max()),score_max_abs=float(delta[:,4:].max())))
            prediction=non_max_suppression(torch.from_numpy(actual),conf_thres=.25,iou_thres=.45,nc=1)[0];prediction[:,:4]=scale_boxes((384,640),prediction[:,:4],(720,1280));detections.append(dict(id=p['samples'][i]['id'],method=name,boxes=prediction.tolist()))
        expected_boxes=model.predict(images[i],**options)[0].boxes;detections.append(dict(id=p['samples'][i]['id'],method='pytorch',boxes=[] if expected_boxes is None else expected_boxes.data.cpu().tolist()))
    def run_one(name,index):
        if name=='pytorch':model.predict(images[index],**options)
        else:
            im=images[index];x=model.predictor.preprocess([im]).cpu().numpy();session=sessions[name];out=session.run(output_names[name],{session.get_inputs()[0].name:x})[0];b=non_max_suppression(torch.from_numpy(out),conf_thres=.25,iou_thres=.45,nc=1)[0];b[:,:4]=scale_boxes((384,640),b[:,:4],im.shape[:2])
    names=['pytorch']+list(sessions)
    for name in names:
        for i in range(PARAMS['warmup']):run_one(name,i%len(images))
    times={name:[] for name in names}
    for repeat in range(PARAMS['repetitions']):
        for i in range(len(images)):
            ordered=names[(repeat+i)%len(names):]+names[:(repeat+i)%len(names)]
            for name in ordered:
                torch.cuda.synchronize();began=time.perf_counter();run_one(name,i);torch.cuda.synchronize();times[name].append((time.perf_counter()-began)*1000)
    for name,session in sessions.items():profile_paths[name]=str(Path(session.end_profiling()).relative_to(OUT))
    provider_nodes={}
    for name,profile in profile_paths.items():
        counts={}
        for r in load(OUT/profile):
            provider=r.get('args',{}).get('provider')
            if provider:counts[provider]=counts.get(provider,0)+1
        provider_nodes[name]=counts
    save(OUT/'benchmark.json',dict(samples=len(images),repetitions=PARAMS['repetitions'],times={name:dict(n=len(v),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),all_ms=v) for name,v in times.items()},raw_parity=raw,detections=detections,failures=failures,providers={name:session.get_providers() for name,session in sessions.items()},provider_node_counts=provider_nodes,profiling_files=profile_paths,scope='Same native input/tensor/FP32/conf/NMS. End-to-end inference only excluding decode/RAW geometry/IPC. CPU ORT shares official preprocessing which currently copies via CUDA; retained as measured, not optimized CPU claim.',truth_guard=True,sources_unchanged=verify()['source_hashes']==p['source_hashes']))
    print('Backend pilot complete',flush=True)
if __name__=='__main__':main()
