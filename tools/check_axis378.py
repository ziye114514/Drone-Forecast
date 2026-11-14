"""Known affine plane motion plus missing-current-observation negatives."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,argparse,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT),str(ROOT/'tools')]
from check_continuation340 import frame
from model_axis_transport import circle_tensor,transform_tensor,ModelAxisTransport


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',default='controls378_20261109');a=p.parse_args()
    out=ROOT.parent/'output'/a.batch;out.mkdir(exist_ok=False);cv2.setNumThreads(2)
    theta=np.deg2rad(5.);A=cv2.getRotationMatrix2D((300.,250.),-5.,1.015)
    first,mask=frame(0);normal=np.array([0.,-np.sqrt(1-(18/55)**2)])
    first.update(native_mask=mask,body=dict(detected=True,center_provisional=False,bbox=[240,230,360,270]),structure=dict(axis_xy=normal.tolist(),axis_support=dict(valid=True,confidence=.9)))
    helper=ModelAxisTransport();helper.step(first,1.);stored=sum(map(np.sum,helper.images))
    second=dict(on=cv2.warpAffine(first['on'],A,(1280,720)),off=cv2.warpAffine(first['off'],A,(1280,720)),
        native_mask=cv2.warpAffine(mask.astype(np.uint8),A,(1280,720),flags=cv2.INTER_NEAREST).astype(bool),
        body=dict(detected=True,center_provisional=False,bbox=[235,225,365,275]),structure={})
    first['on'][:]=0;first['off'][:]=0;assert sum(map(np.sum,helper.images))==stored
    helper.step(second,1+1/30);info=second['structure']['model_axis_transport'];assert info['qualified'],info
    expected=transform_tensor(circle_tensor(normal),A[:,:2])[1];actual=np.asarray(second['structure']['axis_xy'])
    angle=np.rad2deg(np.arccos(np.clip(abs(expected@actual)/(np.linalg.norm(expected)*np.linalg.norm(actual)),-1,1)))
    assert angle<2 and abs(np.linalg.norm(expected)-np.linalg.norm(actual))<.03,(angle,expected,actual)
    missing=dict(second,body=dict(detected=False),structure={});helper.step(missing,1+2/30);assert not missing['structure']['model_axis_transport']['qualified']
    r=dict(known_rotating_plane_angle_error_deg=float(angle),projected_normal_norm_error=float(abs(np.linalg.norm(expected)-np.linalg.norm(actual))),
        current_image_registration_required=True,missing_current_target_does_not_become_pose_observation=True,
        saved_native_image_owned_despite_source_reuse=True,source_counts_not_new_individual_RPM=True)
    (out/'results.json').write_text(json.dumps(r,indent=2),'utf8');print(r)
