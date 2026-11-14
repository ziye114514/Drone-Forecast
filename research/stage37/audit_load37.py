"""Explicit known-class safe load, tensor-only export, exact architecture parity."""
import importlib,json
from model37 import ROOT,OUT,sha


def main():
    import numpy as np,torch,ultralytics,yaml
    from ultralytics import YOLO
    audit=json.loads((OUT/'serialization_audit.json').read_text(encoding='utf-8'));path=OUT/'upstream/yolo/best.pt'
    assert sha(path)==audit['weight_sha256'];allowed=[];review=[]
    for name in audit['globals_requiring_explicit_allowlist']:
        module,member=name.rsplit('.',1)
        assert (module.startswith('torch.nn.modules.') or module.startswith('ultralytics.nn.modules.') or module in ['ultralytics.nn.tasks','ultralytics.utils','ultralytics.utils.loss','ultralytics.utils.tal']
            or name in ['numpy.dtype','numpy._core.multiarray.scalar']),name
        obj=getattr(importlib.import_module(module),member)
        assert isinstance(obj,type) or name=='numpy._core.multiarray.scalar',name
        allowed.append(obj);review.append(name)
    allowed.extend([type(np.dtype(np.float64)),type(np.dtype(np.float32))])
    with torch.serialization.safe_globals(allowed):checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    network=(checkpoint.get('ema') or checkpoint.get('model')).float().eval();architecture=network.yaml
    # JSON roundtrip rejects executable YAML objects/custom tags.
    architecture=json.loads(json.dumps(architecture,allow_nan=False))
    assert int(architecture['nc'])==len(network.names)
    (OUT/'architecture.yaml').write_text(yaml.safe_dump(architecture,sort_keys=False),encoding='utf-8')
    fresh=YOLO(str(OUT/'architecture.yaml'),task='detect',verbose=False);fresh.model.load_state_dict(network.state_dict(),strict=True);fresh.model.eval()
    torch.set_num_threads(2)
    with torch.inference_mode():
        image=torch.zeros((1,3,320,320));old=network(image);new=fresh.model(image)
        old=old[0] if isinstance(old,(tuple,list)) else old;new=new[0] if isinstance(new,(tuple,list)) else new
        delta=float(torch.max(torch.abs(old-new)))
        assert torch.allclose(old,new,atol=1e-4,rtol=1e-4),(delta,'architecture parity differs')
    torch.save({k:v.detach().cpu() for k,v in network.state_dict().items()},OUT/'tensor_state.pt')
    result=dict(upstream_weight_sha256=sha(path),architecture_sha256=sha(OUT/'architecture.yaml'),tensor_state_sha256=sha(OUT/'tensor_state.pt'),
        torch_version=torch.__version__,ultralytics_version=ultralytics.__version__,names=network.names,parameters=sum(p.numel() for p in network.parameters()),
        reviewed_known_classes=review,weights_only_safe_load=True,tensor_only_reconstruction_strict=True,zero_input_parity_max_abs=delta,
        cuda_available=torch.cuda.is_available(),scope='Official installed architecture strict-loaded from tensor state. No arbitrary upstream scripts or pickle functions executed. Zero input only; no dataset outcomes inspected.')
    (OUT/'model_metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='reviewed_known_classes'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
