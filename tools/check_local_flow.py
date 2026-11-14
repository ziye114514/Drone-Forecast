"""Known image displacement and rotation, plus missing-observation fallback."""
from pathlib import Path
import argparse,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_local_flow import FeatureFlow,velocity_coefficient,DESIGN
parser=argparse.ArgumentParser();parser.add_argument('--crop',action='store_true');parser.add_argument('--padding',type=int,default=192)
parser.add_argument('--output-dir',type=Path);args=parser.parse_args()
OUT=args.output_dir.resolve() if args.output_dir else ROOT.parent/'output'/('local_displacement_crop_20261026' if args.crop else 'local_displacement_20261026');OUT.mkdir(exist_ok=False)
cv2.setNumThreads(2);cv2.setRNGSeed(261026)
rng=np.random.default_rng(261026);image=np.zeros((720,1280),np.uint8)
for x,y in rng.integers([500,300],[680,450],size=(90,2)):
    cv2.rectangle(image,(int(x),int(y)),(int(x)+3,int(y)+3),255,-1)
body=dict(detected=True,bbox=[480,280,710,480],center=[595,380],confidence=.9)
checks={}
for name,angle,shift in [('translation',0.,[4.,-3.]),('rotation',2.,[0.,0.])]:
    affine=cv2.getRotationMatrix2D(tuple(body['center']),angle,1.);affine[:,2]+=shift
    second=cv2.warpAffine(image,affine,(1280,720),flags=cv2.INTER_NEAREST)
    moved=dict(body,center=(np.asarray(body['center'])+shift).tolist(),
        bbox=(np.asarray(body['bbox'])+np.tile(shift,2)).tolist())
    observer=FeatureFlow(crop=args.crop,padding=args.padding);observer.step(1.,image,body);result=observer.step(1.04,second,moved)
    assert result['valid'],result
    error=float(np.linalg.norm(np.asarray(result['displacement'])-shift));assert error<.5,(name,error)
    checks[name]=dict(displacement=result['displacement'],known_displacement=shift,error_px=error,inliers=result['inliers'],cropped=args.crop,padding=args.padding)
observer=FeatureFlow();observer.step(1.,image,body)
assert not observer.step(1.04,image,dict(detected=False))['valid']
assert not observer.step(1.08,image,body)['valid']
c=dict(position=[40.,60.],velocity=[100.,-20.],acceleration=[0.,0.],conditional_scale_rate=None)
same,audit=velocity_coefficient(c,np.eye(2)*100.,dict(valid=False,reason='no_flow'),1.)
assert same is c and not audit['applied']
obs=dict(valid=True,velocity=[110.,-30.],velocity_covariance=(np.eye(2)*25.).tolist(),midpoint_t=1.)
changed,audit=velocity_coefficient(c,np.eye(2)*100.,obs,1.)
assert changed['position']==c['position'] and changed['acceleration']==c['acceleration'] and c['velocity']==[100.,-20.]
assert np.linalg.norm(np.asarray(changed['velocity'])-obs['velocity'])<1e-5
checks.update(missing_current_not_fabricated=True,invalid_flow_exact_fallback=True,only_velocity_changes=True)
(OUT/'mechanism_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
(OUT/'design.json').write_text(json.dumps(dict(**DESIGN,primary_references=[
    'https://docs.prophesee.ai/stable/samples/modules/cv/sparse_flow_py.html',
    'https://docs.opencv.org/4.x/dc/d6b/group__video__track.html',
    'https://github.com/opencv/opencv/blob/4.x/samples/python/lk_track.py'],
    scope='Fixed current v1 detections at exact native cutoffs, no future/GT for extraction; not production RAW integration.'),indent=2),encoding='utf-8')
print(json.dumps(checks),flush=True)
