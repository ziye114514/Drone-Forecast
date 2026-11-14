import json,os,shutil,time
from common44 import *
os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_CONFIG_DIR']=str(ROOT.parent/'tmp/yolo_config44');os.environ['TORCH_HOME']=str(ROOT.parent/'tmp/torch44');os.environ['CUDA_CACHE_PATH']=str(ROOT.parent/'tmp/cuda_cache44')
def main():
    p=verify();weight=ROOT.parent/'output/stage40_record_split_finetune_20261018/training/weights/best.pt';assert sha(weight)==PARAMS['weight_sha256']
    target=OUT/'weights/best.pt';target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();shutil.copyfile(weight,target);assert sha(target)==sha(weight)
    import torch,ultralytics,onnx,onnxruntime
    from ultralytics import YOLO
    torch.set_num_threads(2);model=YOLO(str(target),task='detect');began=time.perf_counter()
    exported=Path(model.export(format='onnx',imgsz=PARAMS['export_shape'],opset=17,batch=1,device='cpu',dynamic=False,simplify=False,nms=False,quantize=32,verbose=False));assert exported.resolve().is_relative_to(OUT.resolve())
    graph=onnx.load(exported);onnx.checker.check_model(graph)
    save(OUT/'export.json',dict(model=str(exported.relative_to(OUT)),sha256=sha(exported),source_weight_sha256=sha(weight),wall_s=time.perf_counter()-began,torch=torch.__version__,ultralytics=ultralytics.__version__,onnx=onnx.__version__,ort=onnxruntime.__version__,available_providers=onnxruntime.get_available_providers(),input_shape=[d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim],sources_unchanged=verify()['source_hashes']==p['source_hashes']))
    print('ONNX artifact verified',exported,flush=True)
if __name__=='__main__':main()
