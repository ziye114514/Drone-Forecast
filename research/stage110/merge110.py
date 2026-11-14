"""Preserve independently measured pre-closing components when closing joins them.

Standard morphology / component provenance; no new-method originality claim.
No forced rotor count, hidden disk, frequency or body-center observation.
"""
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage61')]
from preline61 import PrelinePaper
from recovery import RecoveryFrontend

PARAMS=dict(modes=['closed','open_only','merge_safe'],minimum_child_parent_area=.15,
    minimum_children_coverage=.7,minimum_child_quality=.45,maximum_child_size_ratio=2.5,
    minimum_child_ratio=.12,minimum_child_minor_px=3.,
    source='Native20 neighbor ON/OFF normalized to10ms. Original global threshold and2x2 opening retained. Original3x3 closing retained unless it joins at least two independently qualifying open components.',
    qualification='Direct least squares ellipse only to qualify broad pre-closing components. Original line-first PaperPerception actually fits selected contours and performs grouping/recovery. No template-completed points.',
    scope='Classical morphology repair / ablation, not a claimed novel ellipse detector or independent motor identity. All104 conic/5-frame lock thresholds unchanged.')

class MergeRecovery(RecoveryFrontend,PrelinePaper):
    def __init__(self,w,h,mode='merge_safe'):
        super().__init__(w,h,'neighbor10');assert mode in PARAMS['modes'];self.merge_mode=mode
        self.preline_audit=dict(thin_routed=0,thin_rejected=0,line_measured=0);self.merge_audit={}

    def _eligible(self,c,activity,b,parent_area):
        if len(c)<6:return False
        area=cv2.contourArea(c);x,y,w,h=cv2.boundingRect(c)
        if area<max(self.cfg.min_blob_area,PARAMS['minimum_child_parent_area']*parent_area):return False
        if area>self.cfg.max_blob_fraction*activity.size or max(w,h)*b<10:return False
        try:(_, _),(a,d),_=cv2.fitEllipseDirect(c.astype(np.float32))
        except cv2.error:return False
        major=max(a,d)*b;minor=min(a,d)*b;fill=area/max(np.pi*a*d/4,1.)
        if not(10<=major<=min(self.w*.5,max(w,h)*b*1.5)):return False
        return bool(minor>=PARAMS['minimum_child_minor_px'] and minor/max(major,1)>=PARAMS['minimum_child_ratio']
            and .4<fill<1.6 and np.exp(-abs(fill-1))>=PARAMS['minimum_child_quality']
            and activity[y:y+h,x:x+w].sum()>=30)

    def contours_before_merge(self,contours,activity,b):
        density=cv2.GaussianBlur(activity,(3,3),.8);positive=density[density>0]
        threshold=max(2.,float(np.quantile(positive,self.cfg.density_quantile))) if len(positive) else 2.
        raw=np.uint8(density>=threshold)*255
        opened=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
        children,_=cv2.findContours(opened,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        audit=dict(mode=self.merge_mode,threshold=threshold,closed_components=len(contours),
            open_components=len(children),split_parents=0,retained_children=0,no_fixed_rotor_count=True)
        if self.merge_mode=='open_only':self.merge_audit=audit;return children
        parent_map=np.zeros(activity.shape,np.int32)
        for i,c in enumerate(contours):cv2.drawContours(parent_map,[c],-1,i+1,-1)
        groups={}
        for child in children:
            x,y=child[0,0];parent=int(parent_map[y,x])-1
            if parent>=0:groups.setdefault(parent,[]).append(child)
        selected=[]
        for i,parent in enumerate(contours):
            area=cv2.contourArea(parent);qualified=[c for c in groups.get(i,[]) if self._eligible(c,activity,b,area)]
            sizes=[max(cv2.boundingRect(c)[2:]) for c in qualified]
            good=len(qualified)>=2 and sum(cv2.contourArea(c) for c in qualified)>=PARAMS['minimum_children_coverage']*area
            good=good and max(sizes,default=1)/max(min(sizes,default=1),1)<=PARAMS['maximum_child_size_ratio']
            if good:
                selected.extend(qualified);audit['split_parents']+=1;audit['retained_children']+=len(qualified)
            else:selected.append(parent)
        self.merge_audit=audit;return selected

    def _candidates(self,contours,activity,t,b,thin_only=False):
        if not thin_only and self.merge_mode!='closed':contours=self.contours_before_merge(contours,activity,b)
        elif not thin_only:self.merge_audit=dict(mode='closed',split_parents=0,retained_children=0)
        return super()._candidates(contours,activity,t,b,thin_only)

    def detect_prepared(self,*args,**kwargs):
        self.preline_audit=dict(thin_routed=0,thin_rejected=0,line_measured=0)
        return super().detect_prepared(*args,**kwargs)
