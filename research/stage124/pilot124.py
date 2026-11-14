"""Development-only mask inspection. No future labels, no efficacy claim."""
from pathlib import Path
import json
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT.parent/'output/stage124_occupancy_support_20261023'

def main():
    OUT.mkdir(exist_ok=True)
    panels=[];records=[]
    for stamp in ['43.50','45.00']:
        z=np.load(ROOT.parent/f'output/stage123_frozen_extension_20261023/extraction_trace2/{stamp}_counts.npz')
        on,off=z['on'],z['off'];both=((on>0)&(off>0)).astype(np.uint8)
        smooth=cv2.GaussianBlur(both.astype(np.float32),(3,3),.8)
        masks=[('native ON/OFF',None),('old count >=2',z['closed']),('binary bipolar',both*255),
            ('binary close3',cv2.morphologyEx(both*255,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))),
            ('occupancy local >=.35',np.uint8(smooth>=.35)*255),('occupancy local >=.5',np.uint8(smooth>=.5)*255)]
        row=[]
        for title,m in masks:
            if m is None:
                im=np.full((*on.shape,3),255,np.uint8);im[on>0]=(80,80,240);im[off>0]=(220,180,20);im[both>0]=0
            else:
                im=cv2.cvtColor(m,cv2.COLOR_GRAY2BGR);contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                info=[]
                for c in contours:
                    if len(c)<6 or cv2.contourArea(c)<20:continue
                    ell=cv2.fitEllipse(c);cv2.ellipse(im,ell,(0,0,255),1)
                    info.append(dict(area=cv2.contourArea(c),ellipse=ell))
                records.append(dict(stamp=stamp,mask=title,nonzero=int(np.count_nonzero(m)),components=info))
            im=cv2.resize(im,(504,239),interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(im,(0,0),(504,24),(240,240,240),-1);cv2.putText(im,stamp+' '+title,(5,17),cv2.FONT_HERSHEY_SIMPLEX,.43,(20,20,20),1)
            row.append(im)
        panels.extend([np.hstack(row[:3]),np.hstack(row[3:])])
    cv2.imwrite(str(OUT/'pilot_masks.png'),np.vstack(panels));(OUT/'pilot_masks.json').write_text(json.dumps(records,indent=2),'utf-8')
if __name__=='__main__':main()
