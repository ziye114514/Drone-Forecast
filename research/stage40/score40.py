"""Current visible-box localization; not a trajectory or 6D-pose benchmark."""
import numpy as np
from common40 import *

def iou(a,b):
    a=np.array(a);b=np.array(b);wh=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]));inter=float(np.prod(wh));return inter/max(1e-9,float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter))
def main():
    verify();assert sha(OUT/'predictions.jsonl')==load(OUT/'inference.json')['predictions_sha256'];prep=load(OUT/'preparation.json');assert sha(OUT/'targets.json')==prep['targets_sha256']
    truth={r['id']:r for r in load(OUT/'targets.json')['targets']};details=[]
    for r in map(json.loads,(OUT/'predictions.jsonl').read_text(encoding='utf-8').splitlines()):
        t=truth[r['id']];boxes=sorted(r['boxes'],key=lambda b:-b['confidence']);d={k:r[k] for k in ['id','method','recording','split']};d.update(target_t=t['target_t'],state=t['state'],detected=bool(boxes),boxes=len(boxes))
        if boxes:d['bbox']=boxes[0]['bbox'];d['confidence']=boxes[0]['confidence']
        if t['state']=='present' and boxes:
            a=np.array(boxes[0]['bbox']);b=np.array(t['bbox']);error=float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2));d.update(error_px=error,iou=iou(a,b),center_success=error<=.25*np.linalg.norm(b[2:]-b[:2]),iou_success=iou(a,b)>=.5,any_box_iou_success=any(iou(x['bbox'],b)>=.5 for x in boxes))
        details.append(d)
    table=[]
    for n in [3,5,6]:
        for method in ['pretrained','finetuned']:
            dd=[x for x in details if x['recording']==n and x['method']==method];pp=[x for x in dd if x['state']=='present'];aa=[x for x in dd if x['state']=='absent'];hit=[x for x in pp if x['detected']]
            table.append(dict(recording=n,method=method,present=len(pp),detected_present=len(hit),center_success=sum(x.get('center_success',False) for x in pp),iou_success=sum(x.get('iou_success',False) for x in pp),any_box_iou_success=sum(x.get('any_box_iou_success',False) for x in pp),mean_error_px=float(np.mean([x['error_px'] for x in hit])) if hit else None,absent=len(aa),false_positive_frames=sum(x['detected'] for x in aa)))
    save(OUT/'analysis.json',dict(table=table,details=details,scope=PARAMS['scope']));print(json.dumps(table,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
