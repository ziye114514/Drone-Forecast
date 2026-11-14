"""Nonexecuting checkpoint-global inventory; no upstream application is imported."""
from pathlib import Path
import hashlib,json,os

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT.parent/'output/stage37_upstream_detector_20261018'
os.environ['TORCH_HOME']=str(ROOT.parent/'tmp/torch37')
os.environ['YOLO_CONFIG_DIR']=str(ROOT.parent/'tmp/yolo_config37')
os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_OFFLINE']='true'


def main():
    import torch
    path=OUT/'upstream/yolo/best.pt';assets=json.loads((OUT/'upstream_assets.json').read_text(encoding='utf-8'))
    expected=next(x['sha256'] for x in assets['files'] if x['path']=='yolo/best.pt');actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==expected
    globals_=torch.serialization.get_unsafe_globals_in_checkpoint(path)
    result=dict(weight_sha256=actual,torch_version=torch.__version__,cuda_available=torch.cuda.is_available(),
        globals_requiring_explicit_allowlist=sorted(globals_),checkpoint_loaded=False,
        scope='torch serialization static inventory only; review classes before safe loading. No detector inference or upstream scripts executed.')
    (OUT/'serialization_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
