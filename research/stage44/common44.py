"""Fixed native-input comparison of existing PyTorch and ONNX Runtime backends."""
from pathlib import Path
import datetime,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage43'),str(ROOT/'research/stage41')]
import common41
OUT=ROOT.parent/'output/stage44_onnx_backend_20261019';INPUT=common41.OUT
PARAMS=dict(export_shape=[384,640],opset=17,precision='float32',dynamic=False,simplify=False,nms_export=False,batch=1,
    confidence=.25,iou_nms=.45,threads=2,warmup=5,repetitions=5,
    weight_sha256='ed11fada477ca14ebd4ed0e34a6e73f52cd6112222dbe2941335aa917ee49988',
    baseline='PyTorch original40, original imgsz640/rect=True native1280x720 gives tensor1x3x384x640. ONNX fixed384x640 uses identical letterbox, FP32 and official NMS, no input downscale beyond existing network preprocessing.',
    selection='Twelve fixed frames: each of clear6/side6/train3/negative5 at1/6,1/2,5/6 of frozen41 sequence. No labels/GT chosen. Timing rotates three backends in same process after warmup, batch1; also records raw tensor/detection difference and actual EP availability.')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sources():
    result=common41.sources()
    for name in ['common44.py','export44.py','benchmark44.py']:result['research/stage44/'+name]=sha(Path(__file__).with_name(name))
    return result
def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];common41.verify();assert sha(INPUT/'inputs.json')==p['input_manifest_sha256'];return p
def freeze():
    assert not (OUT/'protocol.json').exists();rows=load(INPUT/'inputs.json')['images'];samples=[]
    for case in ['clear6','side6','train3','negative5']:
        rr=[r for r in rows if r['case']==case]
        for frac in [1/6,.5,5/6]:samples.append(rr[min(len(rr)-1,int(len(rr)*frac))])
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parameters=PARAMS,input_manifest_sha256=sha(INPUT/'inputs.json'),samples=samples,
        design='Reuse installed official Ultralytics ONNX exporter and ONNX Runtime1.30 CUDA/CPU, no installs or architecture changes. Export a copied trusted local40 weight, never write into frozen40 directory. Check EP actually activated; record failures without silent CPU-as-GPU claims. No labels in export/benchmark.',
        references=['https://docs.ultralytics.com/modes/export/','https://docs.ultralytics.com/modes/predict/','https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html']))
    print('Stage44 backend pilot frozen',flush=True)
if __name__=='__main__':freeze()
