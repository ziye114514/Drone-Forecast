import argparse,datetime,importlib.util,os,sys,time
from backend46 import *
os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_CONFIG_DIR']=str(ROOT.parent/'tmp/yolo_config44');os.environ['CUDA_CACHE_PATH']=str(ROOT.parent/'tmp/cuda_cache44')
SEQ=OUT/'sequence'
def convert():
    p=verify();assert not MODEL.exists();MODEL.parent.mkdir(parents=True,exist_ok=True)
    import onnx
    from onnxruntime.transformers import float16
    began=time.perf_counter();original=onnx.load(common44.OUT/'weights/best.onnx');converted=float16.convert_float_to_float16(original,keep_io_types=True,min_positive_val=1e-7,max_finite_val=1e4);onnx.checker.check_model(converted);onnx.save(converted,MODEL)
    from collections import Counter
    save(OUT/'conversion.json',dict(sha256=sha(MODEL),bytes=MODEL.stat().st_size,original_sha256=p['original_onnx_sha256'],wall_s=time.perf_counter()-began,initializer_types=dict(Counter(str(r.data_type) for r in converted.graph.initializer)),input_element_type=converted.graph.input[0].type.tensor_type.elem_type,output_element_type=converted.graph.output[0].type.tensor_type.elem_type,sources_unchanged=verify()['source_hashes']==p['source_hashes']));print('FP16 model converted and checked',flush=True)
def benchmark():
    verify();assert not (OUT/'benchmark.json').exists();path=ROOT/'research/stage45/benchmark45.py';source=path.read_text(encoding='utf-8');a="for kind in ['run','binding','graph']:";assert source.count(a)==1;source=source.replace(a,"for kind in ['fp32_graph','fp16_graph']:")
    namespace={'__name__':'paired_precision46','__file__':str(path)};exec(compile(source,str(path),'exec'),namespace)
    namespace.update(OUT=OUT,verify=verify,Backend=lambda kind:graph45.Backend('graph') if kind=='fp32_graph' else fp16_backend('graph'),boxes=finite_boxes)
    import torch
    with torch.inference_mode():namespace['main']()
def freeze_sequence():
    verify();assert not (SEQ/'protocol.json').exists();parent=load(graph45.OUT/'sequence/protocol.json');metadata=dict(parent)
    metadata.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parent46_protocol_sha256=sha(OUT/'protocol.json'),model_sha256=sha(MODEL),
        design='All1550 native41 frames, one fixed FP16 graph candidate, no sample/threshold changes; use unchanged45 sequence runner and41 replay/scoring with only Backend factory and output folder substituted. Original geometry/public pretrained reference cached. All new neural predictions saved first, all raw outputs finite. No new blind data.')
    save(SEQ/'protocol.json',metadata);print('FP16 full sequence frozen',flush=True)
def verify_sequence():
    verify();p=load(SEQ/'protocol.json')
    for name,value in p['source_hashes'].items():assert sha(ROOT/name)==value
    assert sha(common44.INPUT/'inputs.json')==p['input_manifest_sha256'];assert sha(MODEL)==p['model_sha256'];return p
def sequence_action(action):
    verify_sequence();path=ROOT/'research/stage45/sequence45.py';spec=importlib.util.spec_from_file_location('fp16_sequence46',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.SEQ=SEQ;module.OUT=OUT;module.verify_seq=verify_sequence;module.Backend=fp16_backend;module.boxes=finite_boxes;getattr(module,action)()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['convert','benchmark','freeze_sequence','infer','replay','score']);a=p.parse_args()
    if a.action in ['infer','replay','score']:sequence_action(a.action)
    else:globals()[a.action]()
