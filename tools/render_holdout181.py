"""Reference review images use all original ON/OFF events, no model output."""
from pathlib import Path
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from pure_visibility170 import window
BASE=ROOT.parent/'output/motion181_20261107'

if __name__=='__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for n in [15,19]:
        out=BASE/f'reference_review{n}';out.mkdir(exist_ok=True)
        times=np.arange(10.8,16.0001,.2);manifest=[]
        for t in times:
            e=window(BASE/f'cache_recording{n}',t);e=e[e['t']>=round(t*1e6)-20000]
            ids=e['y'].astype(np.int64)*1280+e['x']+e['p'].astype(np.int64)*1280*720
            counts=np.bincount(ids,minlength=2*1280*720).reshape(2,720,1280)
            rgb=np.zeros((720,1280,3),np.uint8)
            rgb[:,:,0]=np.uint8(np.clip(np.log1p(counts[1])*50,0,255));rgb[:,:,2]=np.uint8(np.clip(np.log1p(counts[0])*50,0,255))
            p=out/f'{t:.1f}.png';cv2.imwrite(str(p),rgb[:,:,::-1]);manifest.append(dict(t=float(t),file=p.name,events=len(e)))
        (out/'manifest.json').write_text(json.dumps(dict(recording=n,model_outputs_used=False,images=manifest),indent=2),'utf-8')
        fig,axes=plt.subplots(3,2,figsize=(14,12))
        for ax,t in zip(axes.flat,[10.8,11.8,12.8,13.8,14.8,15.8]):
            im=cv2.cvtColor(cv2.imread(str(out/f'{t:.1f}.png')),cv2.COLOR_BGR2RGB)
            ax.imshow(im);ax.set_title(f'Recording{n} RAW {t:.1f}s');ax.set_xlim(0,1280);ax.set_ylim(720,0)
        fig.tight_layout();fig.savefig(out/'montage.png',dpi=120);plt.close(fig)
        print(out,flush=True)
