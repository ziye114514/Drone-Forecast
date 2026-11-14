"""Export the validation-selected fixed5epoch checkpoint, preserving parents."""
from pathlib import Path
import hashlib,json,os,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(ROOT/'.cache/ultralytics'))


def main(output):
    import torch,onnx
    from ultralytics import YOLO
    torch.set_num_threads(2);out=Path(output)/'native33_adaptation'
    complete=json.loads((out/'train_completion.json').read_text());source=out/'training/weights/best.pt'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==complete['best_sha256']
    dest=out/'export';dest.mkdir(exist_ok=False);weight=dest/'best.pt';shutil.copyfile(source,weight)
    exported=Path(YOLO(str(weight),task='detect').export(format='onnx',imgsz=[384,640],opset=17,
        batch=1,device='cpu',dynamic=False,simplify=False,nms=False,quantize=32,verbose=False))
    graph=onnx.load(exported);onnx.checker.check_model(graph)
    shape=lambda v:[d.dim_value for d in v.type.tensor_type.shape.dim]
    assert shape(graph.graph.input[0])==[1,3,384,640] and shape(graph.graph.output[0])==[1,5,5040]
    result=dict(model=str(exported),sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),
        source_weight_sha256=complete['best_sha256'],trained_FRED=[3,4,5],validation_FRED=[6],
        source_window_us=33333,epochs=5,test_trajectories_used_for_checkpoint_selection=False)
    (out/'export.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main(sys.argv[1])
