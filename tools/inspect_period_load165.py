"""Determine native NDFT work and burst redundancy before changing it."""
from pathlib import Path
import json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1]
out=ROOT.parent/'output/native_plane165_20261104';records=[]
for clip in ['clear','side']:
    for index in [0,8,14,23]:
        with np.load(ROOT.parent/f'output/rotor_pipeline_20261101/cache/{clip}/{index:04d}.npz') as z:e=z['events'];t=float(z['t'])
        end=round(t*1e6);begin=end-33333;e=e[(e['t']>=begin)&(e['t']<end)]
        p=e['y'].astype(np.int64)*1280+e['x'];nt=np.bincount(p[e['t']-begin<16666.5],minlength=1280*720)
        total=np.bincount(p,minlength=1280*720).reshape(720,1280).astype(np.int32)
        local=cv2.boxFilter(total,cv2.CV_32S,(31,31),normalize=False,borderType=cv2.BORDER_REPLICATE)
        eligible=(total>=6)&(total.astype(np.int64)*961>=2*local)&(nt.reshape(720,1280)>=6)&((total-nt.reshape(720,1280))>=6)
        keep=eligible.reshape(-1)[p];selected=e[keep];pixels=p[keep];n=len(selected)
        counts=total[eligible];keys=(pixels*33333+(selected['t']-begin))*2+selected['p']
        unique=len(np.unique(keys));order=np.lexsort((selected['t'],pixels));pid=pixels[order];pol=selected['p'][order]
        runs=int(np.count_nonzero(np.r_[True,(pid[1:]!=pid[:-1])|(pol[1:]!=pol[:-1])])) if n else 0
        records.append(dict(clip=clip,index=index,raw_events=len(e),eligible_pixels=int(eligible.sum()),
            NDFT_events=n,events_per_pixel_mean=float(counts.mean()) if len(counts) else 0,
            events_per_pixel_max=int(counts.max()) if len(counts) else 0,
            exact_timestamp_polarity_duplicates=n-unique,polarity_runs=runs))
(out/'period_load.json').write_text(json.dumps(records,indent=2),'utf-8');print(json.dumps(records))
