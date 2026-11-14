import json
import numpy as np
from common37 import ROOT,OUT,PARAMS,load,save,sha,verify


def iou(a,b):
    a=np.asarray(a);b=np.asarray(b);area=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
    return float(area/max(np.prod(np.maximum(a[2:]-a[:2],0))+np.prod(np.maximum(b[2:]-b[:2],0))-area,1e-9))


def main():
    p=verify();info=load(OUT/'preparation.json');assert sha(OUT/'targets.json')==info['targets_sha256']
    assert sha(OUT/'predictions.jsonl')==load(OUT/'inference.json')['predictions_sha256']
    for name,value in p['label_hashes'].items():assert sha(ROOT.parent/'output/raw_only_project/evaluation_data'/name)==value
    targets={r['id']:r for r in load(OUT/'targets.json')['targets']};details=[];table=[]
    for r in map(json.loads,(OUT/'predictions.jsonl').read_text(encoding='utf-8').splitlines()):
        t=targets[r['id']];top=max(r['boxes'],key=lambda b:b['confidence']) if r['boxes'] else None
        d=dict(id=r['id'],case=r['case'],format=r['format'],window_us=r['window_us'],target_state=t['state'],detected=bool(top),boxes=len(r['boxes']),target_t=t['target_t'],cutoff_us=r['cutoff_us'])
        if t['state']=='present' and top is not None:
            b=np.array(t['bbox']);error=float(np.linalg.norm((np.array(top['bbox'][:2])+top['bbox'][2:])/2-(b[:2]+b[2:])/2));diag=float(np.linalg.norm(b[2:]-b[:2]))
            d.update(center_error_px=error,top_iou=iou(top['bbox'],t['bbox']),top_joint_center=bool(error<=diag*.25),any_iou50=any(iou(x['bbox'],t['bbox'])>=.5 for x in r['boxes']),top_box=top)
        details.append(d)
    for case in p['cases']:
        for kind in PARAMS['formats']:
            for window in PARAMS['windows_us']:
                dd=[r for r in details if r['case']==case and r['format']==kind and r['window_us']==window];present=[r for r in dd if r['target_state']=='present'];detected=[r for r in present if r['detected']];absent=[r for r in dd if r['target_state']=='absent']
                table.append(dict(case=case,format=kind,window_us=window,present=len(present),present_detected=len(detected),top_joint_center=sum(r.get('top_joint_center',False) for r in present),
                    top_iou50=sum(r.get('top_iou',0)>=.5 for r in present),any_iou50=sum(r.get('any_iou50',False) for r in present),
                    conditional_center_mean_px=float(np.mean([r['center_error_px'] for r in detected])) if detected else None,absent=len(absent),absent_published=sum(r['detected'] for r in absent)))
    baseline=[]
    for case in p['cases']:
        pp=[r for r in targets.values() if r['case']==case and r['state']=='present'];errors=[]
        for t in pp:
            o=t.get('baseline_geometry')
            if o is not None and o['detected']:
                b=np.array(t['bbox']);errors.append(float(np.linalg.norm(np.asarray(o['center'])-(b[:2]+b[2:])/2)))
        baseline.append(dict(case=case,present=len(pp),detected=len(errors),conditional_center_mean_px=float(np.mean(errors)) if errors else None))
    save(OUT/'analysis.json',dict(table=table,details=details,baseline_current_geometry=baseline,scope='Top confidence selection uses no truth. Any-IoU is separately labelled oracle diagnostic; no future accuracy claims.'))
    print('Stage37 snapshot detector migration scored',flush=True)


if __name__=='__main__':main()
