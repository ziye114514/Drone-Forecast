"""Known shared affine and unequal rotor energy; no real-flight accuracy claim."""
from pathlib import Path
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from native_plane_motion import estimate_plane_motion

if __name__=='__main__':
    rng=np.random.default_rng(492);h,w=720,1280;y,x=np.mgrid[:h,:w];parts=[]
    texture=cv2.GaussianBlur(rng.uniform(0,1,(h,w)).astype(np.float32),(5,5),1.)
    mask=np.zeros((h,w),bool)
    for cx,cy in [(530,300),(740,300),(540,440),(730,440)]:
        mm=((x-cx)/55)**2+((y-cy)/25)**2<=1;mask|=mm
        parts.append(mm*(10+60*texture))
    a=sum(parts).astype(np.float32);b=sum(g*p for g,p in zip([.2,4.,2.,.3],parts)).astype(np.float32)
    center=np.array([640.,360.]);T=cv2.getRotationMatrix2D(tuple(center),2.,1.01)
    T[:,2]+=[9.25,-4.4];b=cv2.warpAffine(b,T,(w,h));newmask=cv2.warpAffine(mask.astype(np.uint8),T,(w,h),flags=cv2.INTER_NEAREST)>0
    truth=T[:,:2]@center+T[:,2]-center;zero=np.zeros_like(a);results={}
    for arm in ('control','local'):
        shift,info=estimate_plane_motion((a,zero),mask,(b,zero),newmask,center,
            require_center_support=True,local_contrast=arm=='local')
        results[arm]=dict(qualified=shift is not None,reason=info.get('reason'),
            error_px=None if shift is None else float(np.linalg.norm(np.array([shift['dx'],shift['dy']])-truth)))
    assert results['local']['qualified'],results
    assert results['local']['error_px']<1.,results
    print(json.dumps(dict(truth=truth.tolist(),results=results,real_accuracy_claim=False)))
