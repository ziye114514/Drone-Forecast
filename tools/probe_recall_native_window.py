"""Try otherwise discarded native events before changing or retraining a model.

No annotations in inference. Fixed full-packet33ms, plus an independent crop
arm seeded only by its own current detections. Not a live throughput test.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src')]
for k,v in dict(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',
               YOLO_CONFIG_DIR=str(ROOT/'.cache/ultralytics'),CUDA_CACHE_PATH=str(ROOT/'.cache/cuda')).items():os.environ[k]=v


def main(output):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from event_drone_forecast import detector_backend as module
    from live_roi_refine import refine
    out=Path(output);frames=json.loads((out/'frames.json').read_text('utf-8'))
    model=ROOT/'models/live_position_accumulated_v1.onnx'
    assert hashlib.sha256(model.read_bytes()).hexdigest()=='ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4'
    backend=module.Backend('graph',model_path=model)
    seq=None;history=[];results=[]
    for row in frames:
        if seq!=row['sequence']:seq=row['sequence'];history=[]
        packet=np.load(out/'frames33'/seq/f"{row['index']:04d}.npz")
        image=packet['image'];t=row['t'];start=time.perf_counter()
        b=module.boxes(backend.run(module.preprocess(image)))
        full=[dict(bbox=x[:4].tolist(),confidence=float(x[4])) for x in b]
        full_ms=(time.perf_counter()-start)*1000;candidate=full.copy();diag=dict(attempted=False)
        if not full and history and 0<t-history[-1][0]<=.3:
            hint=history[-1][1].copy();velocity=np.zeros(2)
            if len(history)>1:
                dt=history[-1][0]-history[-2][0]
                if .01<dt<=.3:
                    old,new=history[-2][1],history[-1][1]
                    velocity=((new[:2]+new[2:])-(old[:2]+old[2:]))/(2*dt)
                    if np.linalg.norm(velocity)>1500:velocity[:]=0
            hint+=np.tile(velocity*(t-history[-1][0]),2)
            start=time.perf_counter();found,diag=refine(backend,module,image,hint)
            diag=dict(diag,attempted=True,ms=(time.perf_counter()-start)*1000)
            if found is not None:candidate=[dict(found,recovered=True)]
        if candidate:
            history.append((t,np.asarray(max(candidate,key=lambda x:x['confidence'])['bbox'])))
            history=history[-2:]
        results.append(dict(sequence=seq,index=row['index'],t=t,full33=full,candidate=candidate,zoom=diag,full_frame_ms=full_ms))
        if row['index']==119:
            group=[r for r in results if r['sequence']==seq]
            print(json.dumps(dict(sequence=seq,full33=sum(bool(r['full33']) for r in group),
                with_zoom=sum(bool(r['candidate']) for r in group),zoom_attempts=sum(r['zoom']['attempted'] for r in group))),flush=True)
    (out/'native33_detections.jsonl').write_text('\n'.join(json.dumps(r) for r in results)+'\n','utf-8')


if __name__=='__main__':main(sys.argv[1])
