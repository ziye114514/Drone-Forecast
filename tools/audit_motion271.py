"""Test whether the native match hides a common velocity bias.

Dense fixed-threshold outlines are diagnostic proxies, not independent COM
truth and never runtime inputs here. Past fits cannot see future outlines.
The protocol reuses the existing 20ms/count8/component16 and 0.2s fit.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from replay_joint236 import source
from pure_visibility170 import window
from reference_boxes181 import outline_box
from compare_final_pipeline import reference_at,stats
from physical_publication import evaluate


def fit(history,t):
    h=[r for r in history if 0<=t-r['t']<=.2+1e-8 and r['center'] is not None]
    if len(h)<6:return None
    ts=np.array([r['t'] for r in h])-t
    if -ts[0]<.16-1e-8 or max(np.diff(ts))>.08:return None
    X=np.c_[np.ones(len(ts)),ts,.5*ts*ts];Y=np.array([r['center'] for r in h])
    beta=np.linalg.lstsq(X,Y,rcond=None)[0]
    return dict(position=beta[0].tolist(),velocity=beta[1].tolist(),acceleration=beta[2].tolist(),
        RMS_px=np.sqrt(np.mean((Y-X@beta)**2,axis=0)).tolist(),span_s=float(-ts[0]),future_samples=0)


if __name__=='__main__':
    OUT=ROOT.parent/'output/motion271_20261108';OUT.mkdir(exist_ok=False)
    result=dict(runtime_modified=False,diagnostic_reference_is_not_COM_truth=True,
        reserved_raw_opened=False,weak_reference_thresholds_changed=False,cases={})
    for n in ['13','15','16']:
        rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()]
        cache=ROOT.parent/f'output/{"motion181_20261107" if n=="15" else "phase185_20261107"}/cache_recording{n}'
        history=[];fits=[];review=[]
        for i,r in enumerate(rows):
            e=window(cache,r['t']);e=e[e['t']>=round(r['t']*1e6)-20000]
            ids=e['y'].astype(np.int64)*1280+e['x'];count=np.bincount(ids,minlength=1280*720).reshape(720,1280)
            box=outline_box(count,8);weak=outline_box(count,4)
            center=None if box is None else ((np.array(box[:2])+box[2:])/2).tolist()
            delta=None if box is None or weak is None else float(np.linalg.norm(np.asarray(center)-(np.array(weak[:2])+weak[2:])/2))
            history.append(dict(t=r['t'],box=box,center=center,threshold_center_disagreement_px=delta))
            fits.append(fit(history,r['t']))
            if i%15==0:
                im=np.uint8(255-np.clip(count*20,0,255));im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
                if box is not None:
                    cv2.rectangle(im,tuple(box[:2]),tuple(box[2:]),(0,160,0),1)
                    x,y,X,Y=box;lo=np.maximum(0,[x-30,y-30]);hi=np.minimum([1280,720],[X+30,Y+30]);im=im[lo[1]:hi[1],lo[0]:hi[0]]
                review.append((r['t'],im,delta))
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        refs=sorted(json.loads((ROOT.parent/f'output/{folder}/references.json').read_text('utf8'))['records'].values(),key=lambda r:r['t'])
        records=[]
        for t in np.round(np.arange(10.8,15.6001,.2),1):
            inds=[i for i,r in enumerate(rows) if r['t']<=t+1e-9];i=inds[-1];r=rows[i];f=fits[i];target=reference_at(refs,t+.4)
            if not r['valid'] or target is None or f is None:continue
            h=t-r['t']+.4;pred=np.array(evaluate(r,[h])[str(float(h))]);v=np.array(r['coefficient']['velocity']);vf=np.array(f['velocity'])
            records.append(dict(t=float(t),error_px=float(np.linalg.norm(pred-target)),
                past_outline_velocity_only_error_px=float(np.linalg.norm(pred+h*(vf-v)-target)),
                velocity=np.asarray(v).tolist(),past_outline_fit=f,velocity_difference_px_s=float(np.linalg.norm(vf-v)),
                shape_mode=r['structure']['mode'],threshold_center_disagreement_px=history[i]['threshold_center_disagreement_px']))
        summary=dict(baseline=stats([r['error_px'] for r in records]),
            past_outline_velocity_only=stats([r['past_outline_velocity_only_error_px'] for r in records]),
            velocity_difference=stats([r['velocity_difference_px_s'] for r in records]),
            outline_threshold_difference=stats([r['threshold_center_disagreement_px'] for r in history if r['threshold_center_disagreement_px'] is not None]))
        result['cases'][n]=dict(summary=summary,records=records,dense_proxy=history)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(4,4,figsize=(14,10))
        for ax in axes.flat:ax.axis('off')
        for ax,(t,im,d) in zip(axes.flat,review):
            ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(f'{t:.3f}s d={d:.1f}px' if d is not None else f'{t:.3f}s absent')
        fig.tight_layout();fig.savefig(OUT/f'review_{n}.png',dpi=120);plt.close(fig)
        (OUT/'partial.json').write_text(json.dumps(result,indent=2),'utf8')
        print(n,summary,flush=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),'utf8')
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
