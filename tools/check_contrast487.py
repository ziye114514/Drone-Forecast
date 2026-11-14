"""Known local contrast changes and translation, not real-event force truth."""
from pathlib import Path
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from local_contrast_motion import normalize
from rotor_support_front import estimate_region_translation

if __name__=='__main__':
    rng=np.random.default_rng(487);a=rng.uniform(.2,4,(120,160)).astype(np.float32)
    invariant=float(np.max(abs(normalize(a)-normalize(2.5*a+3))))
    assert invariant<1e-4,invariant
    assert not np.any(normalize(np.ones((32,32),np.float32)))
    h,w=180,320;y,x=np.mgrid[:h,:w];parts=[]
    for cx,cy in [(80,55),(195,55),(90,125),(205,125)]:
        p=np.exp(-.5*(((x-cx)/27)**2+((y-cy)/12)**2))
        p*=1+.25*np.cos(.21*(x-cx))+.2*np.sin(.3*(y-cy));parts.append(p)
    a=40*sum(parts);b=40*sum(s*p for s,p in zip([.2,4.,2.,.3],parts))
    truth=np.array([9.25,-4.4]);T=np.c_[np.eye(2),truth]
    b=cv2.warpAffine(b.astype(np.float32),T,(w,h))
    ma=a>.05;mb=cv2.warpAffine(ma.astype(np.uint8),T,(w,h),flags=cv2.INTER_NEAREST)>0
    zero=np.zeros_like(a);results={}
    for arm in ('control','local'):
        s=estimate_region_translation((a,zero),ma,(b,zero),current_support=mb,
            width=w,height=h,native_matching=True,subpixel=True,interior_border=True,
            centered_matching=True,peak_shape=True,joint_peak=True,peak_covariance=False,
            local_contrast=arm=='local')
        assert s is not None and not s['peak_at_search_boundary']
        results[arm]=dict(xy=[s['dx'],s['dy']],error_px=float(np.linalg.norm(np.array([s['dx'],s['dy']])-truth)))
    print(json.dumps(dict(affine_intensity_descriptor_max_delta=invariant,known_translation=truth.tolist(),
        changed_component_event_energy=True,results=results,real_accuracy_claim=False)))
