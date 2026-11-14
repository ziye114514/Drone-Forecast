"""Small, physically relevant checks for subpixel/estimator bookkeeping."""
from pathlib import Path
import sys,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rotor_support_front import estimate_region_translation
from rotor_thrust_bank import RotorThrustBankProxy
from rotor_force_observer import RotorForceObserver


def check():
    # Synthetic diffuse rotor texture shifted by a known non-integer amount.
    rng=np.random.default_rng(164);image=np.zeros((720,1280),np.float32)
    image[270:380,420:730]=rng.uniform(0,50,(110,310))
    image=cv2.GaussianBlur(image,(9,9),2.)
    support=image>0;delta=np.array([2.3,-1.2])
    matrix=np.array([[1.,0.,delta[0]],[0.,1.,delta[1]]])
    current=cv2.warpAffine(image,matrix,(1280,720))
    shifted=cv2.warpAffine(support.astype(np.uint8),matrix,(1280,720))>0
    args=((image,np.zeros_like(image)),support,(current,np.zeros_like(image)))
    integer=estimate_region_translation(*args,current_support=shifted,native_matching=True)
    precise=estimate_region_translation(*args,current_support=shifted,native_matching=True,subpixel=True)
    old=float(np.linalg.norm(np.array([integer['dx'],integer['dy']])-delta))
    new=float(np.linalg.norm(np.array([precise['dx'],precise['dy']])-delta))
    assert new<.1 and new<old
    # A600Hz local spectrum and300Hz pooled median can be two explanations of
    # unchanged visible activity. Switching estimator is NOT a motor command.
    bank=RotorThrustBankProxy(vector_pool=True);model=RotorForceObserver();n=[0.,-1.]
    transitions=[]
    for i,source in enumerate(['local']*7+['global']*7+['local']*7):
        f=600. if source=='local' else 300.;t=(i+1)/30
        u,q,meta=bank.step(t,dict(mode='edge_on',axis_xy=n),[],[],
            dict(qualified=True,frequency_hz=f,selection_source=source))
        c,v,force,_=model.step(t,dict(detected=True,center=[0.,0.]),u,q,input_meta=meta)
        assert np.linalg.norm(c['acceleration'])<1e-8
        if force.get('visibility_partition_changed'):transitions.append(i)
    return dict(integer_shift_error_px=old,subpixel_shift_error_px=new,
        estimator_switch_no_fake_acceleration=True,partition_transition_ticks=transitions)


if __name__=='__main__':
    result=check();path=ROOT.parent/'output/side_temporal164_20261101/mechanism_checks.json'
    path.write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
