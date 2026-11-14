"""Motion proposal before periodic validation; never a drone detection itself.

Use3x3 coherent occupancy rather than a periodic-pixel count floor, then31px closing
only to propose an ROI (its edges are NEVER fitted as rotor rims). Equal1ms
position-bin means in the earlier half estimate translation. All original
timestamps/polarities survive warp; the existing early/late period test decides
whether a proposed object has enough validated evidence to enter the tracker.
"""
import cv2
import numpy as np
from native_event_alignment import warp


def activity_roi(on,off):
    count=on+off
    occupancy=np.float32(count>0)
    neighbors=cv2.boxFilter(occupancy,-1,(3,3),normalize=False,borderType=cv2.BORDER_CONSTANT)
    mask=(occupancy>0)&(neighbors>=3)
    closed=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((31,31),np.uint8))
    number,labels,stats,_=cv2.connectedComponentsWithStats(closed,8)
    ids=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>=40);ids=ids[ids!=0]
    if not len(ids):return None
    mass=np.bincount(labels.ravel(),weights=count.ravel(),minlength=number)
    best=int(ids[np.argmax(mass[ids])]);x,y,w,h,_=stats[best]
    x=max(0,int(x)-12);y=max(0,int(y)-12);X=min(1280,int(stats[best,0]+w)+12);Y=min(720,int(stats[best,1]+h)+12)
    return [x,y,X,Y]

def propose(events,t,on,off,active=False):
    empty=dict(qualified=False,ROI_is_not_rotor_detection=True,future_events_used=False,
        velocity_uses_earlier_half_only=True,original_timestamps_polarities_retained=True)
    roi=activity_roi(on,off)
    if roi is None:return None,dict(empty,reason='no_coarse_activity_region')
    x,y,X,Y=roi
    end=round(t*1e6);begin=end-33333;mid=begin+33333//2
    keep=(events['t']>=begin)&(events['t']<end)&(events['x']>=x)&(events['x']<X)&(events['y']>=y)&(events['y']<Y)
    e=events[keep];train=e[e['t']<mid]
    if not len(train):return None,dict(empty,reason='empty_discovery_half',roi=[x,y,X,Y])
    bins=((train['t']-begin)//1000).astype(int);n=np.bincount(bins,minlength=17);valid=n>0
    if valid.sum()<8:return None,dict(empty,reason='few_coarse_time_bins',roi=[x,y,X,Y])
    tt=np.bincount(bins,weights=(train['t']-begin)/1e6,minlength=17)[valid]/n[valid]
    xy=np.c_[np.bincount(bins,weights=train['x'],minlength=17)[valid]/n[valid],
             np.bincount(bins,weights=train['y'],minlength=17)[valid]/n[valid]]
    A=np.c_[np.ones(len(tt)),tt-tt.mean()];fit=np.linalg.lstsq(A,xy,rcond=None)[0];velocity=fit[1]
    residual=xy-A@fit
    if not np.isfinite(velocity).all() or np.max(abs(velocity))>160/.033333 or not active and np.linalg.norm(velocity)<6/.1:
        return None,dict(empty,reason='translation_proposal_outside_existing_motion_budget',roi=[x,y,X,Y],velocity_px_s=velocity.tolist())
    result=warp(e,t,velocity,1280,720)
    return result,dict(empty,qualified=True,reason='coarse_translation_hypothesis_pending_period_validation',
        roi=[x,y,X,Y],velocity_px_s=velocity.tolist(),discovery_position_fit_rms_px=float(np.sqrt(np.mean(residual**2))),
        discovery_bins=int(valid.sum()),discovery_events=len(train),source_events=len(e),warped_events=len(result),
        discovery_velocity_source_end_us=mid,source_cutoff_us=end)
