"""Export isolated validation-selected detector and compare its parent on val."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output/fred_domain_adaptation_20261029'
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(OUT/'local_config'))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    import torch,onnx
    from ultralytics import YOLO
    torch.set_num_threads(2)
    completed=json.loads((OUT/'train_completion.json').read_text());design=json.loads((OUT/'protocol.json').read_text())
    weight=OUT/'training/weights/best.pt';assert sha(weight)==completed['best_sha256']
    target=OUT/'export';target.mkdir(exist_ok=False);local=target/'best.pt';shutil.copyfile(weight,local)
    model=YOLO(str(local),task='detect');begin=time.perf_counter()
    exported=Path(model.export(format='onnx',imgsz=[384,640],opset=17,batch=1,device='cpu',
        dynamic=False,simplify=False,nms=False,quantize=32,verbose=False))
    graph=onnx.load(exported);onnx.checker.check_model(graph)
    shape=lambda x:[d.dim_value for d in x.type.tensor_type.shape.dim]
    assert shape(graph.graph.input[0])==[1,3,384,640]
    assert shape(graph.graph.output[0])==[1,5,5040]
    result=dict(model=str(exported),sha256=sha(exported),source_weight_sha256=sha(weight),
        input_shape=shape(graph.graph.input[0]),output_shape=shape(graph.graph.output[0]),
        export_wall_s=time.perf_counter()-begin,original_default_model_unchanged=True,
        FRED_training_sequences=design['training_FRED_sequences'],FRED_validation_sequences=[6])
    (OUT/'export.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result),flush=True)
    parent=YOLO(design['parent_weight'],task='detect')
    m=parent.val(data=str(OUT/'dataset.yaml'),project=str(OUT),name='parent_validation',exist_ok=False,
        device=0,imgsz=640,batch=8,workers=0,rect=True,plots=False,verbose=False,save_json=False)
    values={k:float(v) for k,v in m.results_dict.items()}
    (OUT/'parent_validation.json').write_text(json.dumps(values,indent=2),'utf-8');print(json.dumps(values),flush=True)


if __name__=='__main__':main()
