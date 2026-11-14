"""Exact native20ms reviewed cutoffs; isolate geometry with a fixed saved ROI."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from fast_structure_front import FastStructure,roi_bounds
from run_rotor_only import serial
from research.stage147.curvature147 import notches
OUT=ROOT.parent/'output/geometry170_20261106'

def window(cache,t):
    m=json.loads((cache/'manifest.json').read_text('utf-8'));end=round(t*1e6);begin=end-20000;parts=[]
    for i,r in enumerate(m['frames']):
        prior=m['spec']['start'] if i==0 else m['frames'][i-1]['t']
        if r['t']<begin/1e6 or prior>end/1e6:continue
        if m.get('storage')=='npy':a=np.load(cache/r['file'])
        else:
            with np.load(cache/f'{i:04d}.npz') as z:a=z['events'].copy()
        a=a[(a['t']>=begin)&(a['t']<end)];parts.append(a)
    return np.concatenate(parts)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True);assert not (OUT/'before.json').exists();cv2.setNumThreads(2)
    observations=[json.loads(s) for s in (ROOT.parent/'output/nn_rotor166_20261105/side_observations.jsonl').read_text('utf-8').splitlines()]
    records=[]
    for label,t in [('A',110.02),('B',111.02),('C',112.02),('D',114.02)]:
        e=window(ROOT.parent/'output/side_temporal164_20261101/cache_side_warm',t)
        pair=[]
        for pol in (1,0):
            a=e[e['p']==pol];pair.append(np.bincount(a['y'].astype(np.int32)*1280+a['x'],minlength=1280*720).reshape(720,1280).astype(np.float32))
        available=[r for r in observations if r['t']<=t and r['body'].get('detected') and r['body'].get('bbox') and t-r['t']<.2]
        if not available:
            records.append(dict(label=label,t=t,skipped='no_recent_past_ROI_for_geometry_isolation'));continue
        near=max(available,key=lambda r:r['t'])
        body=near['body'];bounds=roi_bounds(body['bbox']);x,y,X,Y=bounds
        def counter(_e,b,_t,_s):
            u,v,U,V=b;return [a[v:V,u:U] for a in pair]
        model=FastStructure(optimized_compute=True,side_axis_budget=True);model.count_current=counter
        s,r,g,_=model.step(e[:0],t,body,None,np.zeros(2),defer_rates=True)
        on,off=[a[y:Y,x:X] for a in pair];f=np.minimum(cv2.GaussianBlur(on,(3,3),.8),cv2.GaussianBlur(off,(3,3),.8));f[(on+off)<2]=0
        raw=np.uint8(cv2.GaussianBlur(f,(3,3),.8)>=4)*255
        mask=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((2,2),np.uint8));mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cs=sorted([c for c in contours if len(c)>=16 and cv2.contourArea(c)>=20],key=cv2.contourArea,reverse=True)[:12]
        rec=dict(label=label,t=t,body=body,ROI_source_t=near['t'],bounds=bounds.tolist(),structure=s,geometry=g,
                 contours=[dict(points=c.reshape(-1,2).tolist(),area=cv2.contourArea(c),notches=notches(c)) for c in cs],
                 fixed_ROI_from_saved_NN=True,pure_rotor_detection_measured=False)
        records.append(rec);np.savez_compressed(OUT/f'{label}.npz',on=pair[0],off=pair[1],t=t)
        print(label,'mode',s['mode'],'contours',len(cs),'rotors',len(g['rotors']),'sizes',[(round(r['major'],1),round(r['minor'],1),r['mode']) for r in g['rotors']],flush=True)
    (OUT/'before.json').write_text(json.dumps(records,default=serial,indent=2),'utf-8')
