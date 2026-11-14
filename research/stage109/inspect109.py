"""Render captured current-event intermediates; no detector or future targets."""
from pathlib import Path
import json, sys
import cv2
import numpy as np
from scipy.ndimage import maximum_filter, label
sys.path.insert(0,str(Path(__file__).parent))
from audit109 import OUT, save

def main():
    summary=[]
    for file in sorted(OUT.glob('*/*_counts.npz')):
        a=np.load(file);on=a['on'];off=a['off'];mask=a['closed'];origin=a['origin']
        source=json.loads(file.with_name(file.name.replace('_counts.npz','_audit.json')).read_text('utf-8'))
        native=np.full((*on.shape,3),255,np.uint8)
        native[(on>0)&(off==0)]=(40,80,220);native[(off>0)&(on==0)]=(220,170,20);native[(on>0)&(off>0)]=(15,15,15)
        distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
        peaks=(distance==maximum_filter(distance,size=7))&(distance>=2.5)
        components,n=label(peaks);pp=[]
        for k in range(1,n+1):
            y,x=np.nonzero(components==k);j=np.argmax(distance[y,x]);pp.append([int(x[j]),int(y[j]),float(distance[y[j],x[j]])])
        metrics=[]
        for name in ['raw_mask','opened','closed']:
            num,labs,stats,cent=cv2.connectedComponentsWithStats(a[name])
            metrics.append(dict(stage=name,components=[dict(bbox=[int(x+origin[0]),int(y+origin[1]),int(x+w+origin[0]),int(y+h+origin[1])],area=int(area)) for x,y,w,h,area in stats[1:] if area>=10]))
        overlay=native.copy()
        for i,r in enumerate(source['selected']):
            c=np.asarray(r['center'])-origin
            cv2.ellipse(overlay,(tuple(c), (r['major'],r['minor']),r['angle']),(30,160,30),1)
            cv2.putText(overlay,str(i+1),tuple(np.rint(c).astype(int)),0,.4,(10,70,10),1)
        panels=[native,cv2.applyColorMap(np.uint8(np.clip(a['density']/(2*float(a['threshold'])),0,1)*255),cv2.COLORMAP_TURBO),
            cv2.cvtColor(a['raw_mask'],cv2.COLOR_GRAY2BGR),cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),
            cv2.applyColorMap(np.uint8(distance/max(float(distance.max()),1)*255),cv2.COLORMAP_TURBO),overlay]
        for x,y,d in pp:cv2.circle(panels[4],(x,y),2,(255,255,255),1)
        titles=['Native20 ON/OFF','Repeated bipolar event count','Global threshold mask','After morphology','Distance transform + peaks','Actual old selected ellipses']
        grid=np.full((480,1260,3),245,np.uint8)
        for i,(panel,title) in enumerate(zip(panels,titles)):
            scale=min(410/panel.shape[1],205/panel.shape[0]);im=cv2.resize(panel,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
            y=(i//3)*240;x=(i%3)*420;grid[y:y+im.shape[0],x:x+im.shape[1]]=im
            cv2.putText(grid,title,(x+4,y+225),0,.47,(30,40,50),1)
        dest=file.with_name(file.name.replace('_counts.npz','_pipeline.png'));cv2.imwrite(str(dest),grid)
        item=dict(case=file.parent.name,t=source['t'],threshold=source['threshold'],stages=metrics,
            peaks=[dict(x=x+int(origin[0]),y=y+int(origin[1]),distance=d) for x,y,d in pp],
            ranked=len(source['ranked_candidates']),selected=len(source['selected']))
        summary.append(item)
    save(OUT/'intermediate_summary109.json',summary)
    print(json.dumps([dict(case=r['case'],t=r['t'],ranked=r['ranked'],selected=r['selected'],peaks=len(r['peaks'])) for r in summary]),flush=True)

if __name__=='__main__':main()
