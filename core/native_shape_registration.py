"""Fallback pose registration of native rotor occupancy, not blade texture.

OpenCV ECC aligns current/past support geometry both ways. Existing transform
scale/rotation gates and1.5px bidirectional closure are retained. A mask edge
is not advertised as a physical propeller rim or a new rotor detection.
"""
import cv2
import numpy as np


def estimate(previous_mask,current_mask):
    oy,ox=np.nonzero(previous_mask);cy,cx=np.nonzero(current_mask)
    empty=dict(qualified=False,method='bidirectional_native_occupancy_ECC',new_physical_rim_measured=False)
    if min(len(ox),len(cx))<32:return None,dict(empty,reason='few_native_occupancy_pixels')
    lo=np.maximum(0,np.minimum([ox.min(),oy.min()],[cx.min(),cy.min()])-24)
    hi=np.minimum([1280,720],np.maximum([ox.max(),oy.max()],[cx.max(),cy.max()])+25)
    x,y=lo;X,Y=hi
    old=previous_mask[y:Y,x:X].astype(np.float32);new=current_mask[y:Y,x:X].astype(np.float32)
    a=cv2.GaussianBlur(old,(5,5),1.2);b=cv2.GaussianBlur(new,(5,5),1.2)
    prior=np.array([(cx.min()+cx.max()-ox.min()-ox.max())/2,(cy.min()+cy.max()-oy.min()-oy.max())/2])
    seed=np.c_[np.eye(2),prior].astype(np.float32);back=np.c_[np.eye(2),-prior].astype(np.float32)
    options=dict(motionType=cv2.MOTION_AFFINE,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,50,1e-5),inputMask=None,gaussFiltSize=5)
    try:
        score,M=cv2.findTransformECC(a,b,seed,**options);reverse,N=cv2.findTransformECC(b,a,back,**options)
    except cv2.error:return None,dict(empty,reason='occupancy_optimization_failed')
    info=dict(empty,ecc=float(score),reverse_ecc=float(reverse))
    values=np.linalg.svd(M[:,:2],compute_uv=False);angle=np.rad2deg(np.arctan2(M[1,0]-M[0,1],M[0,0]+M[1,1]))
    if min(score,reverse)<.75 or values.min()<.85 or values.max()>1.18 or abs(angle)>15:
        return None,dict(info,reason='weak_shape_or_nonrigid_transform',singular_values=values.tolist(),rotation_deg=float(angle))
    H=np.vstack([M,[0,0,1]]);B=np.vstack([N,[0,0,1]])
    ids=np.linspace(0,len(ox)-1,min(500,len(ox))).astype(int);points=np.c_[ox[ids]-x,oy[ids]-y,np.ones(len(ids))]
    cycled=points@(B@H).T;errors=np.linalg.norm(cycled[:,:2]-points[:,:2],axis=1);cycle=float(np.median(errors))
    if cycle>1.5:return None,dict(info,reason='shape_cycle_not_closed',cycle_median_px=cycle)
    edge=cv2.morphologyEx(old.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
    target_edge=cv2.morphologyEx(new.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
    distance=cv2.distanceTransform((~target_edge).astype(np.uint8),cv2.DIST_L2,3)
    ey,ex=np.nonzero(edge);p=np.c_[ex,ey,np.ones(len(ex))]@M.T;pixel=np.rint(p).astype(int)
    valid=(pixel[:,0]>=0)&(pixel[:,0]<X-x)&(pixel[:,1]>=0)&(pixel[:,1]<Y-y)
    if valid.sum()<32:return None,dict(info,reason='few_current_shape_edges')
    residual=distance[pixel[valid,1],pixel[valid,0]];variance=max(4.,float(np.median(residual**2)),cycle**2)
    global_M=M.copy().astype(float);global_M[:,2]+=lo-global_M[:,:2]@lo
    info.update(qualified=True,reason='current_native_shape_registration',cycle_median_px=cycle,
        span_fraction=1.,displacement_covariance_px2=(np.eye(2)*variance).tolist(),shared_affine=global_M.tolist())
    return dict(shared_affine=global_M.tolist(),displacement_covariance_px2=(np.eye(2)*variance).tolist()),info
