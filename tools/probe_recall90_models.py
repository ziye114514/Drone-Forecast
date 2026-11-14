"""Reuse the already trained small-target model; no training or label input.

The previous per-frame33ms cache and model provenance are fixed before scoring.
"""
from pathlib import Path
import hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics')


def main(output):
    import numpy as np,torch
    torch.set_num_threads(2)
    from event_drone_forecast import detector_backend as module
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    prior=ROOT.parent/'output/rpm_recall_pilot_20261031'
    frames=json.loads((prior/'frames.json').read_text('utf-8'))
    model=ROOT.parent/'output/fred_domain_adaptation_20261029/export/best.onnx'
    digest=hashlib.sha256(model.read_bytes()).hexdigest()
    assert digest=='ffddb548d4441c04f588f3a888109e28be94c6ee4c122b84d57fb1a0b334bc9c'
    protocol=dict(model=str(model),sha256=digest,input='same current native33ms packets',
        confidence=.25,trained_FRED=[3,4,5],validation_FRED=[6],
        development_clips=[0,7,9],target_IoU50_positive_fraction=.9,
        no_label_crops=True,no_new_training=True,not_global_blind_test=True,
        forecast_target='same actual publication time +0.4s, misses retained')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf-8')
    backend=module.Backend('graph',model_path=model);results=[]
    for row in frames:
        packet=np.load(prior/'frames33'/row['sequence']/f"{row['index']:04d}.npz")
        begin=time.perf_counter();b=module.boxes(backend.run(module.preprocess(packet['image'])))
        results.append(dict(sequence=row['sequence'],t=row['t'],index=row['index'],
            boxes=[dict(bbox=r[:4].tolist(),confidence=float(r[4])) for r in b],
            ms=(time.perf_counter()-begin)*1000))
        if row['index']==119:print(row['sequence'],sum(bool(x['boxes']) for x in results if x['sequence']==row['sequence']),flush=True)
    (out/'adapt33.jsonl').write_text('\n'.join(json.dumps(r) for r in results)+'\n','utf-8')


if __name__=='__main__':main(sys.argv[1])
