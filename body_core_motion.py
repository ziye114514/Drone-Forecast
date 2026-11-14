"""Separate central-body support from fast propeller texture in ONE LK pass.

Fixed central disk radius=min(box size)/4, no orientation/head-tail assumption.
At most60 points as before. The body subset must pass existing strict rigid
motion gates; otherwise the combined current-image tracks remain the fallback.
"""
import cv2
import numpy as np
from live_local_flow import reused_similarity
from rigid_center_translation import choose


def points(image,mask,center,radius):
    core=np.zeros_like(mask);cv2.circle(core,tuple(np.rint(center).astype(int)),max(1,int(radius)),255,-1)
    core=cv2.bitwise_and(core,mask)
    p=cv2.goodFeaturesToTrack(image,30,.01,4,mask=core)
    core_count=0 if p is None else len(p)
    if p is None or len(p)<12:
        p=cv2.goodFeaturesToTrack(image,60,.01,4,mask=mask)
        return p,None,dict(core_candidates=core_count,balanced=False)
    outside=cv2.bitwise_and(mask,cv2.bitwise_not(core))
    q=cv2.goodFeaturesToTrack(image,30,.01,4,mask=outside)
    if q is None:return p,np.ones(len(p),bool),dict(core_candidates=len(p),balanced=True)
    return np.concatenate((p,q)),np.r_[np.ones(len(p),bool),np.zeros(len(q),bool)],dict(core_candidates=len(p),balanced=True)


def qualify(previous,current,fb,good,core,center,box):
    info=dict(applied=False,reason='no_twelve_current_body_tracks',extra_LK_passes=0)
    if core is None:return None,info
    keep=good&core
    if keep.sum()<12:return None,info
    similarity=reused_similarity(previous[keep],current[keep],fb[keep],center,center_support=True)
    _,choice=choose(np.median(current[keep]-previous[keep],axis=0),similarity,box)
    if not choice['applied']:return None,dict(info,reason=choice['reason'],current_body_points=int(keep.sum()))
    return keep,dict(info,applied=True,reason='strict_current_body_core_rigid_support',current_body_points=int(keep.sum()))
