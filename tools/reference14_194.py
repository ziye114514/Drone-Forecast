"""Unchanged counts8/4 outline-reference protocol on frozen recording14, independent of forecast outputs.

These are weak native event-outline references, NOT human/COM ground truth.
"""
from pathlib import Path
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from pure_visibility170 import window
from reference_boxes181 import outline_box
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__=='__main__':
    out=ROOT.parent/'output/timing194_20261107/reference14';out.mkdir(exist_ok=False)
    cache=ROOT.parent/'output/phase185_20261107/cache_recording14';records={};images=[]
    for t in np.round(np.arange(10.8,16.0001,.2),1):
        e=window(cache,t);e=e[e['t']>=round(t*1e6)-20000]
        ids=e['y'].astype(np.int64)*1280+e['x'];count=np.bincount(ids,minlength=1280*720).reshape(720,1280)
        b=outline_box(count,8);weak=outline_box(count,4)
        u=None if b is None or weak is None else float(np.linalg.norm((np.array(b[:2])+b[2:]-np.array(weak[:2])-weak[2:])/2))
        records[f'{t:.1f}']=dict(t=float(t),state='present' if b is not None else 'absent',box=b,
            weaker_outline_box=weak,threshold_center_disagreement_px=u,reference_is_independent_human_GT=False,
            forecast_outputs_used=False)
        im=np.full((720,1280,3),245,np.uint8)
        im[e['y'],e['x']]=np.where(e['p'][:,None]>0,np.array([240,80,30]),np.array([40,70,220]))
        cv2.imwrite(str(out/f'{t:.1f}.png'),im)
        if b is not None:
            cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,180,0),1)
            x,y,X,Y=b;lo=np.maximum(0,[x-40,y-40]);hi=np.minimum([1280,720],[X+40,Y+40]);im=im[lo[1]:hi[1],lo[0]:hi[0]]
        images.append((t,cv2.cvtColor(im,cv2.COLOR_BGR2RGB),u))
    for page,start in enumerate(range(0,len(images),9)):
        fig,axes=plt.subplots(3,3,figsize=(12,9))
        for ax in axes.flat:ax.axis('off')
        for ax,(t,im,u) in zip(axes.flat,images[start:start+9]):
            ax.imshow(im);ax.set_title(f'{t:.1f}s outline difference={u:.1f}px' if u is not None else f'{t:.1f}s no dense outline',fontsize=9)
        fig.tight_layout();fig.savefig(out/f'boxes_page{page}.png',dpi=115);plt.close(fig)
    (out/'references.json').write_text(json.dumps(dict(records=records,
        scope='Unmodified counts8/4, connected component area16 event-visible outline; not independent human or COM ground truth',
        predicted_boxes_read=False,assistant_visual_review_pending=True),indent=2),'utf-8')
    print(dict(references=len(records),present=sum(r['state']=='present' for r in records.values()),threshold_disagreement_max=max([r['threshold_center_disagreement_px'] or 0 for r in records.values()])))
