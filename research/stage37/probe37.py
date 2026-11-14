"""Post-hoc implementation parity probe; no threshold tuning or new scoring."""
import importlib,json
import cv2,numpy as np
from model37 import ROOT,OUT,load_detector,sha
from common37 import load,save


def main():
    import torch
    from ultralytics.data.augment import LetterBox
    meta=load(OUT/'model_metadata.json');path=OUT/'upstream/yolo/best.pt';assert sha(path)==meta['upstream_weight_sha256']
    allowed=[]
    for name in meta['reviewed_known_classes']:
        module,member=name.rsplit('.',1);allowed.append(getattr(importlib.import_module(module),member))
    allowed.extend([type(np.dtype(np.float64)),type(np.dtype(np.float32))])
    with torch.serialization.safe_globals(allowed):checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    original=(checkpoint.get('ema') or checkpoint.get('model')).float().eval();fresh=load_detector().model.float().eval();torch.set_num_threads(2);torch.manual_seed(37)
    inputs=load(OUT/'inputs.json')['images'];chosen=[r for r in inputs if r['case']=='clear6' and r['window_us']==50000][:3]
    tensors=[('random_nonzero',torch.rand((1,3,320,320)))]+[(r['format'],None) for r in chosen];result=[]
    for name,tensor in tensors:
        if tensor is None:
            r=next(r for r in chosen if r['format']==name);image=cv2.imdecode(np.fromfile(OUT/r['image'],np.uint8),cv2.IMREAD_COLOR)
            image=LetterBox((640,640),auto=True,stride=32)(image=image);tensor=torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1)[::-1])).float().unsqueeze(0)/255
        with torch.inference_mode():
            a=original(tensor);b=fresh(tensor);a=a[0] if isinstance(a,(tuple,list)) else a;b=b[0] if isinstance(b,(tuple,list)) else b
            delta=float((a-b).abs().max());equal=bool(torch.allclose(a,b,atol=1e-4,rtol=1e-4))
            result.append(dict(input=name,shape=list(tensor.shape),max_abs=delta,allclose=equal,
                original_max_class_score=float(a[:,4:].max()),reconstructed_max_class_score=float(b[:,4:].max())))
            assert equal,(name,delta)
    save(OUT/'nonzero_parity_probe.json',dict(posthoc=True,helper_sha256=sha(__import__('pathlib').Path(__file__)),results=result,
        scope='Same installed official model classes, exact original EMA versus strict tensor reconstruction before NMS. Not accuracy/transfer proof, no thresholds changed.'))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
