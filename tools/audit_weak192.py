"""Locate native-event loss before modifying frequency or spatial gates."""
from pathlib import Path
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from pure_visibility170 import window
from moving_event_proposal import propose
BASE=ROOT.parent/'output';OUT=BASE/'weak192_20261107'
if __name__=='__main__':
    OUT.mkdir(exist_ok=True);results=[];pictures=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for n in (15,16,17,18,19):
        parent=BASE/('motion181_20261107' if n in (15,19) else 'phase185_20261107');cache=parent/f'cache_recording{n}'
        for t in (10.5,12.,13.5,15.):
            e=window(cache,t);end=round(t*1e6);begin=end-33333;mid=begin+33333/2
            pix=e['y'].astype(np.int64)*1280+e['x'];on=np.bincount(pix[e['p']>0],minlength=921600).reshape(720,1280);off=np.bincount(pix[e['p']==0],minlength=921600).reshape(720,1280)
            _,proposal=propose(e,t,on,off,active=True);roi=proposal.get('roi')
            if roi:
                x,y,X,Y=roi;sel=(e['x']>=x)&(e['x']<X)&(e['y']>=y)&(e['y']<Y);ev=e[sel]
            else:ev=e
            p=ev['y'].astype(np.int64)*1280+ev['x'];nt=np.bincount(p[ev['t']<mid],minlength=921600);nv=np.bincount(p[ev['t']>=mid],minlength=921600)
            count=on+off;local=cv2.boxFilter(count.astype(float),-1,(31,31),normalize=False,borderType=cv2.BORDER_REPLICATE)
            elig=(nt>=6)&(nv>=6);dense=(count>=6)&(count*961>=2*local)
            good=elig&dense.ravel();mixed=np.bincount(p[ev['p']>0],minlength=921600);allc=nt+nv;anypix=allc>0
            mono=(mixed==0)|(mixed==allc)
            rec=dict(recording=n,t=t,ROI_proposal_not_truth=roi,events=len(ev),on_fraction=float(np.mean(ev['p']>0)) if len(ev) else None,
                occupied_pixels=int(anypix.sum()),monopolar_pixels=int((mono&anypix).sum()),eligible_two_halves=int(elig.sum()),eligible_after_density=int(good.sum()),
                eligible_bipolar=int((good&~mono).sum()),proposal_velocity=proposal.get('velocity_px_s'))
            results.append(rec)
            if t==13.5:
                im=np.full((720,1280,3),245,np.uint8);ev=e[e['t']>=end-20000];im[ev['y'],ev['x']]=np.where(ev['p'][:,None]>0,[240,80,30],[40,70,220])
                if roi:cv2.rectangle(im,tuple(roi[:2]),tuple(roi[2:]),(0,170,0),1)
                cv2.imwrite(str(OUT/f'raw{n}.png'),im);pictures.append((n,cv2.cvtColor(im,cv2.COLOR_BGR2RGB)))
    (OUT/'loss_audit.json').write_text(json.dumps(results,indent=2),'utf-8')
    fig,axes=plt.subplots(3,2,figsize=(13,10))
    for ax in axes.flat:ax.axis('off')
    for ax,(n,im) in zip(axes.flat,pictures):ax.imshow(im);ax.set_title(f'Recording {n}, 13.5s - raw ON/OFF + activity ROI (not detection)')
    fig.tight_layout();fig.savefig(OUT/'raw_montage.png',dpi=120);plt.close(fig)
    print(json.dumps(results,indent=2),flush=True)
