"""Declared engineering baseline: activity components + temporal association.

No drone classifier and no learned weights. Assumes one main compact moving
target. It provides image position only, never rotor pose or thrust.
"""
from collections import deque
import cv2,numpy as np

class ActivityTracker:
    def __init__(self,width,height):
        self.w,self.h=width,height;self.b=2;self.frames=deque();self.total=None
        self.center=None;self.velocity=np.zeros(2);self.last_seen=None;self.last_size=40.
        self.pending=None;self.streak=0

    def step(self,events,t):
        b=self.b;h,w=self.h//b,self.w//b
        px=(events['y'].astype(int)//b)*w+events['x'].astype(int)//b
        total=np.bincount(px,minlength=h*w).reshape(h,w).astype(np.float32)
        if self.total is None:self.total=np.zeros_like(total)
        self.frames.append((t,total));self.total+=total
        while self.frames and self.frames[0][0]<t-.025:
            self.total-=self.frames.popleft()[1]
        if self.last_seen is not None and t-self.last_seen>.3:
            self.center=None;self.velocity[:]=0;self.pending=None;self.streak=0
        mask=np.uint8(self.total>=3)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
        weights=np.bincount(labels.ravel(),weights=self.total.ravel(),minlength=n)
        candidates=[];predicted=None if self.center is None else self.center+self.velocity*(t-self.last_seen)
        for k in range(1,n):
            x,y,bw,bh,area=stats[k];activity=weights[k]
            if area<6 or activity<40 or max(bw,bh)*b>320 or area>h*w*.04:continue
            center=(np.array([x+bw/2,y+bh/2])*b)
            gap=0. if predicted is None else np.linalg.norm(center-predicted)
            if predicted is not None and gap>max(60.,self.last_size*1.8):continue
            score=np.log1p(activity)+.3*np.log1p(area)-gap/max(25.,self.last_size*.6)
            candidates.append(dict(center=center,bbox=np.array([x,y,x+bw,y+bh])*b,score=score,size=max(bw,bh)*b,activity=activity))
        result=dict(t=t,detected=False,confidence=0.,rotors=[],n_events=len(events),position_source='activity_only',geometry_quality=0.)
        if not candidates:
            if self.center is None:self.streak=0
            return result
        best=max(candidates,key=lambda q:q['score']);center=best['center']
        if self.center is None:
            self.streak=self.streak+1 if self.pending is not None and np.linalg.norm(center-self.pending)<40 else 1
            self.pending=center
            if self.streak<3:return result
        else:
            dt=max(t-self.last_seen,.001)
            self.velocity=.7*self.velocity+.3*np.clip((center-self.center)/dt,-2500,2500)
        self.center=center;self.last_seen=t;self.last_size=best['size']
        result.update(detected=True,confidence=.75,center=center.tolist(),bbox=best['bbox'].tolist(),
            center_covariance_px2=(np.eye(2)*4.).tolist(),activity=float(best['activity']))
        return result
