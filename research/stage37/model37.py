"""Load only a reviewed tensor state into the installed official detector architecture."""
from pathlib import Path
import hashlib,json,os

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT.parent/'output/stage37_upstream_detector_20261018'
os.environ['TORCH_HOME']=str(ROOT.parent/'tmp/torch37');os.environ['CUDA_CACHE_PATH']=str(ROOT.parent/'tmp/cuda_cache37')
os.environ['YOLO_CONFIG_DIR']=str(ROOT.parent/'tmp/yolo_config37');os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_OFFLINE']='true'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def load_detector():
    import torch
    from ultralytics import YOLO
    metadata=json.loads((OUT/'model_metadata.json').read_text(encoding='utf-8'))
    assert sha(OUT/'architecture.yaml')==metadata['architecture_sha256']
    assert sha(OUT/'tensor_state.pt')==metadata['tensor_state_sha256']
    detector=YOLO(str(OUT/'architecture.yaml'),task='detect',verbose=False)
    state=torch.load(OUT/'tensor_state.pt',map_location='cpu',weights_only=True)
    detector.model.load_state_dict(state,strict=True);detector.model.names={int(k):v for k,v in metadata['names'].items()}
    detector.model.eval();return detector
