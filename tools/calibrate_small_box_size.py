"""Fit one global box-size correction using FRED6 validation only.

Centers and confidence are unchanged. No development/test frames or future
trajectory errors are read when fitting. This does not calibrate uncertainty.
"""
from pathlib import Path
import hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src')]
os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics')


def main(output):
    import cv2,numpy as np,torch
    torch.set_num_threads(2)
    from event_drone_forecast import detector_backend as m
    out=Path(output);source=ROOT.parent/'output/fred_domain_adaptation_20261029'
    model=source/'export/best.onnx';digest=hashlib.sha256(model.read_bytes()).hexdigest()
    backend=m.Backend('graph',model_path=model);samples=[]
    for p in sorted((source/'dataset/val/images').glob('fred6_*.png')):
        lab=source/'dataset/val/labels'/(p.stem+'.txt')
        text=lab.read_text().strip()
        if not text:continue
        values=np.array([float(x) for x in text.splitlines()[0].split()[1:]])*np.array([1280,720,1280,720])
        gt=np.r_[values[:2]-values[2:]/2,values[:2]+values[2:]/2]
        boxes=m.boxes(backend.run(m.preprocess(cv2.imread(str(p)))))
        for row in boxes:
            b=row[:4];inter=np.prod(np.maximum(0,np.minimum(b[2:],gt[2:])-np.maximum(b[:2],gt[:2])))
            union=np.prod(b[2:]-b[:2])+np.prod(gt[2:]-gt[:2])-inter
            if inter/max(union,1e-9)>=.5:
                samples.append((values[2:]/(b[2:]-b[:2])).tolist());break
    if len(samples)<20:raise RuntimeError('Insufficient validation matches')
    scale=np.median(samples,axis=0)
    result=dict(model_sha256=digest,fit_sequences=[6],fit_current_matches=len(samples),
        size_scale_xy=scale.tolist(),centers_changed=False,confidence_changed=False,
        fitting_rule='coordinate-wise median true_size/predicted_size, validation IoU>=.5',
        fit_input_window_us=20000,test_trajectory_labels_used=False)
    (out/'box_size_calibration.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result))


if __name__=='__main__':main(sys.argv[1])
