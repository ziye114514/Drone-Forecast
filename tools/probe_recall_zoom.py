"""Same current image/model: full-frame NN vs existing recent-ROI zoom recovery.

This probe reads no annotations or previous system predictions. It is not a
wall-clock benchmark. Each arm is measured on the same cached native image.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.cache/ultralytics')
os.environ['CUDA_CACHE_PATH'] = str(ROOT / '.cache/cuda')


def main(output):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from event_drone_forecast import detector_backend as module
    from live_roi_refine import refine
    out = Path(output)
    frames = json.loads((out / 'frames.json').read_text('utf-8'))
    weights = ROOT / 'models/live_position_accumulated_v1.onnx'
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert digest == 'ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4'
    backend = module.Backend('graph', model_path=weights)
    results = []
    seq = None
    history = []
    for row in frames:
        if seq != row['sequence']:
            seq = row['sequence']
            history = []
        packet = np.load(row['path'])
        t = float(packet['t'])
        start = time.perf_counter()
        boxes = module.boxes(backend.run(module.preprocess(packet['image'])))
        full_ms = (time.perf_counter()-start)*1000
        control = [dict(bbox=b[:4].tolist(), confidence=float(b[4])) for b in boxes]
        candidate = control.copy()
        audit = dict(attempted=False)
        if not control and history and 0 < t-history[-1][0] <= .3:
            hint = history[-1][1].copy()
            velocity = np.zeros(2)
            if len(history) >= 2:
                dt = history[-1][0]-history[-2][0]
                if .01 < dt <= .3:
                    a, b = history[-2][1], history[-1][1]
                    velocity = ((b[:2]+b[2:])-(a[:2]+a[2:]))/(2*dt)
                    if np.linalg.norm(velocity) > 1500:
                        velocity[:] = 0
            delta = velocity * (t-history[-1][0])
            hint += np.tile(delta, 2)
            begin = time.perf_counter()
            found, audit = refine(backend, module, packet['image'], hint)
            audit = dict(audit, attempted=True, ms=(time.perf_counter()-begin)*1000,
                         hint=hint.tolist(), history_source_t=history[-1][0])
            if found is not None:
                candidate = [dict(found, recovered=True)]
        # Only this arm's actual current image detections can seed later crops.
        strong = [b for b in candidate if b['confidence'] >= .45]
        if strong:
            history.append((t, np.asarray(max(strong, key=lambda b: b['confidence'])['bbox'])))
            history = history[-2:]
        result = dict(sequence=seq, index=row['index'], t=t, control=control,
                      candidate=candidate, zoom=audit, full_frame_ms=full_ms)
        results.append(result)
        if row['index'] == 119:
            group = [r for r in results if r['sequence'] == seq]
            print(json.dumps(dict(sequence=seq, full=sum(bool(r['control']) for r in group),
                with_zoom=sum(bool(r['candidate']) for r in group),
                zoom_attempts=sum(r['zoom']['attempted'] for r in group))), flush=True)
    (out / 'zoom_detections.jsonl').write_text('\n'.join(json.dumps(r) for r in results)+'\n', 'utf-8')


if __name__ == '__main__':
    main(sys.argv[1])
