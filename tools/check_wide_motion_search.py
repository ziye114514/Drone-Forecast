"""A search proposal alone must never become an image observation."""
from pathlib import Path
import sys,json
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from run_fast_structure import track


def main():
    cv2.setNumThreads(2);rng=np.random.default_rng(311030)
    image=np.zeros((720,1280),np.uint8)
    image[100:132,100:148]=rng.integers(0,2,(32,48),dtype=np.uint8)*255
    current=np.zeros_like(image);current[100:132,130:178]=image[100:132,100:148]
    body=dict(detected=True,bbox=[100.,100.,148.,132.],center=[124.,116.],confidence=.8)
    old,old_info=track(image,current,body,rigid_motion=True,rigid_box_center=True,inertial_seed=[30.,0.])
    new,info=track(image,current,body,rigid_motion=True,rigid_box_center=True,inertial_seed=[30.,0.],seed_backward=True,wide_motion_search=True)
    assert new is not None,info
    assert np.linalg.norm(np.array(new['center'])-[154.,116.])<.2,(new,info)
    assert info['wide_motion_search']['old_size_displacement_gate_exceeded']
    missing,missing_info=track(image,np.zeros_like(image),body,rigid_motion=True,rigid_box_center=True,inertial_seed=[30.,0.],seed_backward=True,wide_motion_search=True)
    assert missing is None,missing_info
    stable,stable_info=track(image,image,body,rigid_motion=True,rigid_box_center=True,inertial_seed=[0.,0.],seed_backward=True,wide_motion_search=True)
    assert stable is not None and np.linalg.norm(np.array(stable['center'])-body['center'])<.01
    result=dict(old_observation=old,old_diagnostic=old_info,new_observation=new,new_diagnostic=info,
        absent_image_rejected=True,stationary_center_unchanged=True,real_world_accuracy_proved=False)
    out=ROOT.parent/'output/stability_audit_20261031/wide_search_checks.json';out.write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
