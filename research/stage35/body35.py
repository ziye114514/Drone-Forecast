"""Bounded whole-object LK measurements on native event counts, never reconstructed G."""
from pathlib import Path
from collections import deque
import copy
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[2]
PARAMS=dict(accumulation_s=.02,max_corners=80,quality_level=.01,min_corner_distance_px=4.,minimum_points=6,
    lk_window_px=21,lk_levels=3,maximum_lk_error=30.,forward_backward_px=1.5,ransac_px=3.,minimum_inlier_fraction=.6,
    scale_range=[.85,1.2],max_rotation_deg=25.,max_step_px=80.,maximum_extent_ratio=2.5,
    variants={'similarity_short':dict(age_s=.3,fit='similarity'),'similarity_long':dict(age_s=2.,fit='similarity'),'translation_long':dict(age_s=2.,fit='translation')},
    refresh='Features seeded ONLY on independent rotor/geometry detections; never replenished by flow-only frames. No current events => no body publication. Age measured from last geometry, not last flow.')


class EventImages:
    def __init__(self,w,h):self.w=w;self.h=h;self.history=deque()

    @staticmethod
    def render(count):
        image=np.uint8(255*np.clip(np.log1p(count)/np.log(8),0,1))
        return cv2.GaussianBlur(image,(5,5),.8)

    def step(self,events,t):
        pixels=events['y'].astype(np.int32)*self.w+events['x'];current=np.bincount(pixels,minlength=self.w*self.h).reshape(self.h,self.w).astype(np.uint16)
        self.history.append((t,current))
        while self.history and self.history[0][0]<=t-PARAMS['accumulation_s']+1e-8:self.history.popleft()
        total=sum(x for _,x in self.history)
        return self.render(total),self.render(current),current>=2


class BodyTracker:
    def __init__(self,w,h,method):
        self.w=w;self.h=h;self.method=method;self.spec=PARAMS['variants'][method];self.points=None;self.image=None;self.corners=None;self.last_geometry=None;self.initial_extent=None

    def seed(self,image,strong,t,o):
        b=np.rint(o['bbox']).astype(int);lo=np.maximum(b[:2],0);hi=np.minimum(b[2:],[self.w-1,self.h-1]);mask=np.zeros_like(image)
        if np.all(hi>lo):mask[lo[1]:hi[1]+1,lo[0]:hi[0]+1]=255
        mask[~cv2.dilate(strong.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)]=0
        points=cv2.goodFeaturesToTrack(image,maxCorners=PARAMS['max_corners'],qualityLevel=PARAMS['quality_level'],minDistance=PARAMS['min_corner_distance_px'],mask=mask,blockSize=5)
        self.points=points if points is not None and len(points)>=PARAMS['minimum_points'] else None;self.image=image.copy();self.last_geometry=t
        self.corners=np.array([[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]],float);self.initial_extent=float(max(b[2:]-b[:2]))

    def step(self,image,current_image,strong,t,o,current_events):
        if o['detected']:
            self.seed(image,strong,t,o);return copy.deepcopy(o),dict(source='geometry',points=0 if self.points is None else len(self.points))
        audit=dict(source='none',reason='uninitialized')
        if self.points is None:return copy.deepcopy(o),audit
        age=t-self.last_geometry;audit['geometry_age_s']=age
        if age>self.spec['age_s']+1e-8 or current_events==0:
            self.points=None;audit['reason']='age_expired' if age>self.spec['age_s']+1e-8 else 'no_current_events';return copy.deepcopy(o),audit
        kwargs=dict(winSize=(PARAMS['lk_window_px'],)*2,maxLevel=PARAMS['lk_levels'],criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
        target,st,error=cv2.calcOpticalFlowPyrLK(self.image,image,self.points,None,**kwargs)
        good=st.ravel().astype(bool)&np.isfinite(target.reshape(-1,2)).all(axis=1)&(error.ravel()<PARAMS['maximum_lk_error'])
        old=self.points.reshape(-1,2)[good];new=target.reshape(-1,2)[good]
        audit['forward_points']=len(old)
        if len(old)>=PARAMS['minimum_points']:
            back,status,_=cv2.calcOpticalFlowPyrLK(image,self.image,new.reshape(-1,1,2),None,**kwargs)
            inside=(new[:,0]>=2)&(new[:,0]<self.w-2)&(new[:,1]>=2)&(new[:,1]<self.h-2)
            ix=np.rint(np.clip(new,[0,0],[self.w-1,self.h-1])).astype(int)
            good=status.ravel().astype(bool)&np.isfinite(back.reshape(-1,2)).all(axis=1)&inside
            good&=np.linalg.norm(back.reshape(-1,2)-old,axis=1)<=PARAMS['forward_backward_px']
            good&=current_image[ix[:,1],ix[:,0]]>=8
            old,new=old[good],new[good]
        audit['consistent_points']=len(old);matrix=None;inlier=None
        if len(old)>=PARAMS['minimum_points']:
            if self.spec['fit']=='similarity':
                matrix,inlier=cv2.estimateAffinePartial2D(old,new,method=cv2.RANSAC,ransacReprojThreshold=PARAMS['ransac_px'],maxIters=1000,confidence=.99,refineIters=10)
                if inlier is not None:inlier=inlier.ravel().astype(bool)
            else:
                delta=np.median(new-old,axis=0);inlier=np.linalg.norm(new-old-delta,axis=1)<=PARAMS['ransac_px'];matrix=np.column_stack([np.eye(2),delta])
        valid=matrix is not None and inlier is not None and int(inlier.sum())>=PARAMS['minimum_points'] and float(inlier.mean())>=PARAMS['minimum_inlier_fraction']
        if valid:
            scale=float(np.linalg.norm(matrix[:,0]));angle=float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0])));corners=self.corners@matrix[:,:2].T+matrix[:,2]
            extent=float(max(np.ptp(corners,axis=0)));step=float(np.linalg.norm(corners.mean(0)-self.corners.mean(0)))
            valid=bool(np.isfinite(corners).all() and PARAMS['scale_range'][0]<=scale<=PARAMS['scale_range'][1] and abs(angle)<=PARAMS['max_rotation_deg'] and step<=PARAMS['max_step_px']
                and 1/PARAMS['maximum_extent_ratio']<=extent/max(self.initial_extent,1)<=PARAMS['maximum_extent_ratio'])
            audit.update(scale=scale,rotation_deg=angle,step_px=step,inliers=int(inlier.sum()))
        if not valid:
            self.points=None;audit['reason']='flow_or_transform_rejected';return copy.deepcopy(o),audit
        self.points=np.ascontiguousarray(new[inlier].reshape(-1,1,2),dtype=np.float32);self.image=image.copy();self.corners=corners
        center=corners.mean(0);lo=np.clip(corners.min(0),[0,0],[self.w-1,self.h-1]);hi=np.clip(corners.max(0),[0,0],[self.w-1,self.h-1])
        result=dict(t=t,detected=True,center=center.tolist(),bbox=np.r_[lo,hi].tolist(),confidence=.5,rotors=[],geometry_quality=0.,observed_rotor_count=0,
            source='body_flow',independent_rotor_measurement=False,n_events=current_events)
        audit.update(source='body_flow',reason='accepted_current_feature_motion',remaining_points=len(self.points))
        return result,audit
