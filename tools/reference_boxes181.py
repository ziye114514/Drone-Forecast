"""Blinded image-outline references, explicitly NOT independent human GT.

All native ON/OFF events, fixed counts4/8 and component area16. No predicted
position/box, frequency, rotor hypothesis or forecast is read. Review overlays
must be inspected; threshold disagreement is retained as reference uncertainty.
"""
from pathlib import Path
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from pure_visibility170 import window
BASE=ROOT.parent/'output/motion181_20261107'


def outline_box(count,level):
    binary=np.uint8(count>=level)
    number,ids,stats,_=cv2.connectedComponentsWithStats(binary,8)
    good=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>=16);good=good[good!=0]
    if not len(good):return None
    ss=stats[good];return [int(ss[:,0].min()),int(ss[:,1].min()),int((ss[:,0]+ss[:,2]).max()),int((ss[:,1]+ss[:,3]).max())]


if __name__=='__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for n in [15]:
        out=BASE/f'reference_review{n}';records={};crops=[]
        for item in json.loads((out/'manifest.json').read_text('utf-8'))['images']:
            t=item['t'];e=window(BASE/f'cache_recording{n}',t);e=e[e['t']>=round(t*1e6)-20000]
            ids=e['y'].astype(np.int64)*1280+e['x'];count=np.bincount(ids,minlength=1280*720).reshape(720,1280)
            b=outline_box(count,8);weak=outline_box(count,4)
            uncertainty=None if b is None or weak is None else float(np.linalg.norm((np.array(b[:2])+b[2:]-np.array(weak[:2])-weak[2:])/2))
            record=dict(t=t,state='present' if b is not None else 'absent',box=b,
                weaker_outline_box=weak,threshold_center_disagreement_px=uncertainty,
                reference_is_independent_human_GT=False,forecast_outputs_used=False)
            records[f'{t:.1f}']=record
            im=cv2.imread(str(out/item['file']))
            if b is not None:
                cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,255,0),1)
                x,y,X,Y=b;pad=40;lo=np.maximum(0,[x-pad,y-pad]);hi=np.minimum([1280,720],[X+pad,Y+pad]);im=im[lo[1]:hi[1],lo[0]:hi[0]]
            crops.append((t,cv2.cvtColor(im,cv2.COLOR_BGR2RGB),uncertainty))
        for page,start in enumerate(range(0,len(crops),9)):
            fig,axes=plt.subplots(3,3,figsize=(12,9))
            for ax in axes.flat:ax.axis('off')
            for ax,(t,im,u) in zip(axes.flat,crops[start:start+9]):
                ax.imshow(im);ax.set_title(f'{t:.1f}s outline Δ={u:.1f}px' if u is not None else f'{t:.1f}s no dense outline',fontsize=10)
            fig.tight_layout();fig.savefig(out/f'boxes_page{page}.png',dpi=120);plt.close(fig)
        (out/'references.json').write_text(json.dumps(dict(records=records,
            scope='fixed native event-visible outline centre, not calibratedCOM or independent human bounding boxes',
            reference_method='all original ONOFF counts8, connected native components>=16 pixels; counts4 disagreement retained',
            predicted_boxes_read=False,assistant_visual_review_pending=True),indent=2),'utf-8')
        print(n,len(records),[(k,r['threshold_center_disagreement_px']) for k,r in records.items() if (r['threshold_center_disagreement_px'] or 0)>6])
