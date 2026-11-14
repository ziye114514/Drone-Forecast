"""One declared spatial timing model, no threshold/parameter sweeps."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from regional_timing_period import measure,PARAMS
from moving_event_proposal import propose
from pure_visibility170 import window
OUT=ROOT.parent/'output/timing193_20261107'
if __name__=='__main__':
    OUT.mkdir(exist_ok=False);results=[]
    for n in (15,18):
        parent='motion181_20261107' if n==15 else 'phase185_20261107';cache=ROOT.parent/f'output/{parent}/cache_recording{n}'
        for t in (10.5,12.,13.5,15.):
            e=window(cache,t);pix=e['y'].astype(np.int64)*1280+e['x'];on=np.bincount(pix[e['p']>0],minlength=921600).reshape(720,1280);off=np.bincount(pix[e['p']==0],minlength=921600).reshape(720,1280)
            aligned,proposal=propose(e,t,on,off,active=True)
            for arm,ev in [('raw',e),('aligned',aligned)]:
                if ev is None:continue
                start=time.perf_counter();mask,diag,groups=measure(ev,t);elapsed=time.perf_counter()-start
                rec=dict(recording=n,t=t,arm=arm,diagnostic=diag,groups=groups,elapsed_ms=elapsed*1000,proposal=proposal);results.append(rec)
                im=np.full((720,1280,3),245,np.uint8);q=ev[ev['t']>=round(t*1e6)-20000];im[q['y'],q['x']]=np.where(q['p'][:,None]>0,[240,80,30],[40,70,220])
                for g in groups:
                    x,y=g['cell_xy'];cv2.rectangle(im,(x,y),(x+16,y+16),(0,170,0),1)
                cv2.imwrite(str(OUT/f'{n}_{t}_{arm}.png'),im)
                print(n,t,arm,len(groups),diag.get('maximum_score'),round(elapsed*1000,1),flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(parameters=PARAMS,parameters_searched=False,records=results),indent=2),'utf-8')
