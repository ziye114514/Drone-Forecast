import collections,json
import cv2,numpy as np
from common38 import OUT,OLD,PARAMS,load,save,sha,verify
from score37 import iou


def fuse(boxes):
    if not boxes:return []
    rects=[[b['bbox'][0],b['bbox'][1],b['bbox'][2]-b['bbox'][0],b['bbox'][3]-b['bbox'][1]] for b in boxes]
    indices=cv2.dnn.NMSBoxes(rects,[b['confidence'] for b in boxes],PARAMS['confidence'],PARAMS['iou_nms']);return [boxes[int(i)] for i in np.asarray(indices).ravel()]


def main():
    p=verify();assert sha(OUT/'predictions.jsonl')==load(OUT/'inference.json')['predictions_sha256'];assert sha(OLD/'targets.json')==p['eval_target_sha256'];targets={r['id']:r for r in load(OLD/'targets.json')['targets']}
    rows=list(map(json.loads,(OUT/'predictions.jsonl').read_text(encoding='utf-8').splitlines()));old={(r['id'],r['window_us']):r for r in map(json.loads,(OLD/'predictions.jsonl').read_text(encoding='utf-8').splitlines()) if r['format']=='sdk_default'}
    groups=collections.defaultdict(list)
    for r in rows:
        r['method']=f'n{int(r["noise"])}_d{r["divisor"]}_w{r["window_us"]}';groups[r['id']].extend(r['boxes'])
        if not r['noise'] and r['divisor']==1:assert r['boxes']==old[r['id'],r['window_us']]['boxes']
    for ident,bb in groups.items():rows.append(dict(id=ident,case=targets[ident]['case'],method='fused_all',boxes=fuse(bb)))
    details=[];table=[]
    for r in rows:
        t=targets[r['id']];top=max(r['boxes'],key=lambda b:b['confidence']) if r['boxes'] else None;d=dict(id=r['id'],case=r['case'],method=r['method'],state=t['state'],detected=bool(top),boxes=r['boxes'])
        if top is not None and t['state']=='present':
            b=np.array(t['bbox']);error=float(np.linalg.norm((np.array(top['bbox'][:2])+top['bbox'][2:])/2-(b[:2]+b[2:])/2))
            d.update(center_error_px=error,top_joint_center=bool(error<=np.linalg.norm(b[2:]-b[:2])*.25),top_iou=iou(top['bbox'],t['bbox']),any_iou50=any(iou(x['bbox'],t['bbox'])>=.5 for x in r['boxes']))
        details.append(d)
    methods=sorted(set(d['method'] for d in details))
    for case in p['cases']:
        for method in methods:
            dd=[d for d in details if d['case']==case and d['method']==method];present=[d for d in dd if d['state']=='present'];valid=[d for d in present if d['detected']];absent=[d for d in dd if d['state']=='absent']
            table.append(dict(case=case,method=method,present=len(present),detected=len(valid),top_joint_center=sum(d.get('top_joint_center',False) for d in present),top_iou50=sum(d.get('top_iou',0)>=.5 for d in present),any_iou50=sum(d.get('any_iou50',False) for d in present),
                conditional_mean_px=float(np.mean([d['center_error_px'] for d in valid])) if valid else None,absent=len(absent),absent_published=sum(d['detected'] for d in absent)))
    save(OUT/'analysis.json',dict(table=table,details=details,unfiltered_scale1_matches_parent=True,
        scope='Confidence-ranked fixed-view fusion before truth, no threshold tuning. All individual views retained. Counts across repeated views are not independent samples. No future forecasting evaluated.'))
    print('Noise/scale pilot scored',flush=True)


if __name__=='__main__':main()
