"""Same masked real motion inputs, no lease aliasing or lost rotor evidence."""
from pathlib import Path
import sys,json,time
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from native_history_buffer import NativeHistoryBuffer
from rotor_support_front import estimate_region_translation
from native_plane_motion import _image

if __name__=='__main__':
    cv2.setNumThreads(2);folder=ROOT.parent/'output/native_plane165_20261104';buffer=NativeHistoryBuffer();records=[]
    for clip in ['clear','side']:
        for index in [8,14]:
            with np.load(folder/f'native_maps/{clip}_{index:04d}.npz') as a,np.load(folder/f'native_maps/{clip}_{index+1:04d}.npz') as b:
                previous=(a['on'].copy(),a['off'].copy());mask=a['native'].copy();current=(b['on'],b['off']);cm=b['native']
            timing={}
            for sparse in [False,True,True,False]:
                tick=time.perf_counter()
                if sparse:owned,om=buffer.update(previous,mask,cv2.boundingRect(mask.view(np.uint8)))
                else:owned=tuple(p.copy() for p in previous);om=mask.copy()
                timing.setdefault(str(sparse),[]).append((time.perf_counter()-tick)*1000)
            for p,q in zip(previous,buffer.planes):
                np.testing.assert_array_equal(p[mask],q[mask]);assert not np.any(q[~mask])
            for border in [False,True]:
                kwargs=dict(current_support=cm,native_matching=True,subpixel=True,interior_border=border)
                old=estimate_region_translation(previous,mask,current,**kwargs)
                new=estimate_region_translation(buffer.planes,buffer.mask,current,**kwargs)
                assert old==new
            np.testing.assert_array_equal(_image(previous,mask,(0,0,1280,720)),_image(buffer.planes,buffer.mask,(0,0,1280,720)))
            snapshot=buffer.planes[0].copy();previous[0].fill(0);np.testing.assert_array_equal(snapshot,buffer.planes[0])
            records.append(dict(clip=clip,index=index,native_pixels=int(mask.sum()),same_motion=True,same_plane_image=True,source_overwrite_safe=True,
                ms={k:float(np.median(v)) for k,v in timing.items()}))
    (folder/'native_history_checks.json').write_text(json.dumps(records,indent=2),'utf-8');print(json.dumps(records))
