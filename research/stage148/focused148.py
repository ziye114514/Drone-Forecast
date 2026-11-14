"""Post-inference same-frame comparison, never input to prediction or score."""
from pathlib import Path
import json,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT.parent/'output/stage148_temporal_rotors_20261024'
def rows(path):return list(map(json.loads,path.read_text().splitlines()))
new=rows(OUT/'clear_joint/predictions.jsonl');index=min(range(len(new)),key=lambda i:abs(new[i]['t']-26.84));r=new[index]
old=rows(ROOT.parent/'output/stage147_curvature_split_20261024/clear_paired/predictions.jsonl')[index]
cap=cv2.VideoCapture(str(OUT/'clear_joint/predictions.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=cap.read();cap.release();assert ok
native=im[:,:1280].copy();new_rotors=r['rotor_branch']['geometry']['temporal148']['raw_rotors']
lo=np.maximum(np.asarray(r['rotor_branch']['roi'][:2],int)-8,0);hi=np.minimum(np.asarray(r['rotor_branch']['roi'][2:],int)+8,[1280,720])
tiles=[]
for title,rotors in [('Same ON/OFF display',[]),('147: previous fitted candidates',old['rotor_branch']['geometry']['observed_rotors']),('148: split + merge CURRENT arcs',new_rotors)]:
    display=native.copy()
    for q in rotors:
        guided=q.get('split_method148') in ('prior_guided_current_arcs','prior_guided_fragment_merge');color=(20,180,20) if guided else (20,130,230)
        center=tuple(np.rint([q['cx'],q['cy']]).astype(int));axes=tuple(np.maximum(1,np.rint([q['major']/2,q['minor']/2]).astype(int)))
        cv2.ellipse(display,center,axes,q['angle'],0,360,color,1);cv2.circle(display,center,2,color,-1)
        if guided:
            for point in q['original_arc_points']:cv2.circle(display,tuple(np.rint(point).astype(int)),1,(255,70,0),-1)
    crop=display[lo[1]:hi[1],lo[0]:hi[0]];scale=min(2.5,580/crop.shape[1],320/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
    tile=np.full((390,600,3),255,np.uint8);tile[55:55+crop.shape[0],10:10+crop.shape[1]]=crop
    cv2.putText(tile,title,(10,25),0,.55,(20,35,50),1);tiles.append(tile)
cv2.imwrite(str(OUT/'arcs_guided_comparison.png'),np.concatenate(tiles,axis=1))
print(json.dumps(dict(t=r['t'],guided_arcs=sum(q.get('split_method148')=='prior_guided_current_arcs' for q in new_rotors))),flush=True)
