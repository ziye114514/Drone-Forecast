"""Keep official in-place NMS inside torch inference context; no algorithm change."""
import datetime
from graph45 import *
def main():
    verify();supplement=OUT/'inference_context_supplement.json';assert not supplement.exists() and not (OUT/'benchmark.json').exists()
    save(supplement,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=sha(Path(__file__)),reason='Initial45 passed setup/raw/stale checks but stopped before timed loop: tensor-ready PyTorch forward returned an inference tensor, then official in-place NMS ran outside its context. Wrap whole unchanged benchmark in torch.inference_mode so official NMS may modify it. No inputs/thresholds/weights/timing order change; inference is already intended and no gradients needed.'))
    import torch,benchmark45
    with torch.inference_mode():benchmark45.main()
if __name__=='__main__':main()
