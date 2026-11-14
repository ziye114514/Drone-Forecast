"""Freeze inference first, then score one validation-selected model once."""
from pathlib import Path
import json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(ROOT/'.cache/ultralytics'))


def main(output):
    import cv2,numpy as np,torch
    from event_drone_forecast import detector_backend as module
    torch.set_num_threads(2);out=Path(output);previous=ROOT.parent/'output/rpm_recall_pilot_20261031'
    exported=json.loads((out/'native33_adaptation/export.json').read_text())
    backend=module.Backend('graph',model_path=exported['model']);rows=[]
    for r in json.loads((previous/'frames.json').read_text()):
        image=np.load(previous/'frames33'/r['sequence']/f"{r['index']:04d}.npz")['image']
        b=module.boxes(backend.run(module.preprocess(image)))
        rows.append(dict(sequence=r['sequence'],t=r['t'],index=r['index'],boxes=[dict(bbox=x[:4].tolist(),confidence=float(x[4])) for x in b]))
    (out/'new33_detections.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n','utf-8')
    # Annotation access starts only after the complete inference stream is saved.
    from prepare_fred_transfer import parse
    from score_fred_transfer import reference
    from score_recall_windows import iou
    paths={'FRED0':'fred_transfer_20261029/input/0/coordinates.txt',
        'FRED7':'fred_transfer_20261029/unseen7/input/7/coordinates.txt',
        'FRED9':'stability_audit_20261031/holdout9/input/coordinates.txt'};result={}
    for name,path in paths.items():
        labels=parse((ROOT.parent/'output'/path).read_text('utf-8-sig'));values=[]
        for r in rows:
            if r['sequence']!=name:continue
            box=reference(labels,r['t'])
            if box is None:continue
            best=max(r['boxes'],key=lambda b:b['confidence']) if r['boxes'] else None
            values.append(0. if best is None else iou(best['bbox'],box))
        result[name]=dict(positive_ticks=len(values),IoU50=sum(x>=.5 for x in values),
            fraction=sum(x>=.5 for x in values)/len(values))
    negative=[]
    # Ignore the20us trailing partial packet; only60 complete33.333ms windows.
    for k in range(60):
        im=cv2.imread(str(out/'empty_control'/f'{k:04d}.png'))
        b=module.boxes(backend.run(module.preprocess(im)));negative.append(len(b))
    result['empty_self5']=dict(full_windows=60,frames_with_detections=sum(n>0 for n in negative),
        independent_negative_sequences=1,not_a_global_false_positive_rate=True)
    (out/'new33_scores.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main(sys.argv[1])
