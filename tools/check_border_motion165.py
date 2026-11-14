"""Known shift at a sensor boundary; no future images enter an estimator."""
from pathlib import Path
import sys,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from rotor_support_front import estimate_region_translation
if __name__=='__main__':
    rng=np.random.default_rng(165);h,w=240,384;on=np.zeros((h,w),np.float32);off=on.copy()
    mask=np.zeros((h,w),bool);mask[2:62,100:300]=True
    on[mask]=rng.integers(1,100,mask.sum());off[mask]=rng.integers(1,100,mask.sum())
    transform=np.float32([[1,0,3],[0,1,-4]])
    current=tuple(cv2.warpAffine(a,transform,(w,h)) for a in (on,off))
    cm=cv2.warpAffine(mask.astype(np.uint8),transform,(w,h)).astype(bool);result={}
    for option in [False,True]:
        r=estimate_region_translation((on,off),mask,current,width=w,height=h,current_support=cm,
            native_matching=True,subpixel=True,interior_border=option)
        result[str(option)]=r
    assert abs(result['False']['dy'])<1e-6
    np.testing.assert_allclose([result['True']['dx'],result['True']['dy']],[3,-4],atol=.3)
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
