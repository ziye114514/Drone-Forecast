"""Post-inference localization scoring; no target crops enter either detector."""
from pathlib import Path
import json
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from score_fred_transfer import reference
from prepare_fred_transfer import parse


def iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    inter=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
    return float(inter/max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter,1e-9))


def main(output):
    out=Path(output);data=[json.loads(x) for x in (out/'zoom_detections.jsonl').read_text().splitlines()]
    newer=[json.loads(x) for x in (out/'native33_detections.jsonl').read_text().splitlines()]
    base=ROOT.parent/'output';paths=dict(FRED0=base/'fred_transfer_20261029/input/0/coordinates.txt',
        FRED7=base/'fred_transfer_20261029/unseen7/input/7/coordinates.txt',
        FRED9=base/'stability_audit_20261031/holdout9/input/coordinates.txt')
    truth={k:parse(p.read_text('utf-8-sig')) for k,p in paths.items()}
    result={};changes=[]
    for seq in paths:
        rows=[]
        for a,b in zip(data,newer):
            if a['sequence']!=seq:continue
            gt=reference(truth[seq],a['t'])
            if gt is None:continue
            arms={k:v for k,v in [('full20',a['control']),('full33',b['full33']),('full33_zoom',b['candidate'])]}
            score={}
            for arm,boxes in arms.items():
                best=max(boxes,key=lambda x:x['confidence']) if boxes else None
                score[arm]=dict(output=best is not None,iou=iou(best['bbox'],gt) if best else 0.,
                    center_error=None if not best else float(np.linalg.norm((np.array(best['bbox'][:2])+best['bbox'][2:])/2-(gt[:2]+gt[2:])/2)))
            rows.append(dict(t=a['t'],scores=score))
            if (score['full20']['iou']>=.5)!=(score['full33']['iou']>=.5):
                changes.append(dict(sequence=seq,index=a['index'],t=a['t'],bbox=gt.tolist(),old=a['control'],new=b['full33'],scores=score))
        result[seq]=dict(positive_ticks=len(rows))
        for arm in ['full20','full33','full33_zoom']:
            valid=[r['scores'][arm] for r in rows if r['scores'][arm]['output']]
            result[seq][arm]=dict(outputs=len(valid),IoU50=sum(r['scores'][arm]['iou']>=.5 for r in rows),
                IoU50_fraction=sum(r['scores'][arm]['iou']>=.5 for r in rows)/len(rows),
                center_median_px=float(np.median([r['center_error'] for r in valid])) if valid else None)
        result[seq]['rows']=rows
    (out/'window_scores.json').write_text(json.dumps(result,indent=2),'utf-8')
    selected=[x for x in changes if x['scores']['full33']['iou']>=.5][:6]
    selected += [x for x in changes if x['scores']['full20']['iou']>=.5][:3]
    panels=[]
    for row in selected:
        line=[]
        for folder,key in [('frames','old'),('frames33','new')]:
            image=np.load(out/folder/row['sequence']/f"{row['index']:04d}.npz")['image'].copy()
            b=np.array(row['bbox']);lo=np.maximum(0,np.floor(b[:2]-55)).astype(int);hi=np.minimum([1280,720],np.ceil(b[2:]+55)).astype(int)
            for boxes,color in [([dict(bbox=row['bbox'])],(0,255,0)),(row[key],(255,0,255))]:
                for det in boxes:
                    x,y,X,Y=np.rint(det['bbox']).astype(int);cv2.rectangle(image,(x,y),(X,Y),color,1)
            crop=image[lo[1]:hi[1],lo[0]:hi[0]];crop=cv2.resize(crop,(360,240),interpolation=cv2.INTER_NEAREST)
            panel=np.full((275,370,3),245,np.uint8);panel[30:270,5:365]=crop
            title=f"{row['sequence']} {row['t']:.3f}s {key} IoU={row['scores']['full20' if key=='old' else 'full33']['iou']:.2f}"
            cv2.putText(panel,title,(5,20),0,.45,(20,20,20),1);line.append(panel)
        panels.append(np.hstack(line))
    if panels:cv2.imwrite(str(out/'window_changes.png'),np.vstack(panels))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='rows'} for k,v in result.items()},indent=2))


if __name__=='__main__':main(sys.argv[1])
