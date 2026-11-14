"""Measurement statistic test, independent of any drone trajectory labels."""
from pathlib import Path
import sys,json,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from native_correlation import normalized_search

if __name__=='__main__':
    rng=np.random.default_rng(181);shape=(23,27);original=rng.uniform(0,6,shape).astype(np.float32)
    patch=original-original.mean();target=rng.uniform(0,1,(100,130)).astype(np.float32)
    target[10:33,12:39]=original+40
    target[60:83,80:107]=original+rng.normal(0,.5,shape)
    old,oo,_=normalized_search(target,patch,factor=1,centered=False)
    new,no,_=normalized_search(target,patch,factor=1,centered=True)
    legacy=np.array(np.unravel_index(np.argmax(old),old.shape))+oo
    fixed=np.array(np.unravel_index(np.argmax(new),new.shape))+no
    assert tuple(fixed)==(10,12) and tuple(legacy)!=(10,12)
    shifted,so,_=normalized_search(target+50,patch,factor=1,centered=True)
    peak=np.array(np.unravel_index(np.argmax(shifted),shifted.shape))+so
    assert np.array_equal(peak,fixed)
    result=dict(expected_yx=[10,12],legacy_peak_yx=legacy.tolist(),centered_peak_yx=fixed.tolist(),
        affine_count_offset_invariance=True,parameters_or_trajectory_labels_used=False)
    path=ROOT.parent/'output/motion181_20261107/mechanism.json';path.write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
