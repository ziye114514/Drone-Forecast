"""Past-calibrated periodic line center, only for a short detector gap.

The search box is a proposal, never an observation. Only a qualified current
native periodic line can produce a weak center. This does not recover a box,
rotor identities, RPM, attitude or thrust.
"""
from collections import deque
import numpy as np

PARAMS=dict(calibration_window_s=.35,minimum_pairs=5,maximum_strong_gap_s=.5,
    maximum_offset_MAD_px=6.,minimum_axis_cosine=float(np.cos(np.deg2rad(20))),
    maximum_extent_ratio=2.,weak_measurement_sigma_px=12.,parameters_swept=False)

class PeriodicRecovery:
    def __init__(self,w,h,measure):
        self.w,self.h,self.measure=w,h,measure
        self.pairs=deque();self.last=None;self.last_t=None

    def observe(self,q,body,t,velocity):
        if self.last_t is not None:assert t>self.last_t
        self.last_t=t
        while self.pairs and t-self.pairs[0][0]>PARAMS['calibration_window_s']+1e-8:self.pairs.popleft()
        strong=bool(body.get('detected') and not body.get('low_confidence',False) and body.get('confidence',0)>=.45)
        info=dict(used=False,reason='not_evaluated',search_box_is_measurement=False,
                  mechanical_RPM_estimated=False,thrust_estimated=False)
        if strong:
            self.last=dict(t=t,body=body.copy(),velocity=np.asarray(velocity,float).copy())
            d,_=self.measure(q,body,t,velocity,self.w,self.h)
            if d.get('axis_valid'):
                line=d['line']; offset=np.asarray(body['center'])-np.asarray(line['center'])
                self.pairs.append((t,offset,float(line['angle']),float(line['major'])))
                info.update(reason='calibrated_current_line',pairs=len(self.pairs))
            else:info.update(reason='strong_body_without_current_line')
            return None,d,info
        if body.get('detected'):
            info['reason']='weak_NN_retained';return None,None,info
        if self.last is None or t-self.last['t']>PARAMS['maximum_strong_gap_s']+1e-8:
            self.pairs.clear();info['reason']='no_recent_strong_NN';return None,None,info
        dt=t-self.last['t']; shift=dt*self.last['velocity'];b=np.asarray(self.last['body']['bbox'])+np.tile(shift,2)
        if np.any(b[:2]<0) or np.any(b[2:]>[self.w,self.h]):
            info['reason']='search_outside_sensor';return None,None,info
        proposal=dict(detected=True,bbox=b.tolist(),confidence=1.)
        d,_=self.measure(q,proposal,t,self.last['velocity'],self.w,self.h)
        info.update(search_bbox=b.tolist(),last_strong_age_s=dt,pairs=len(self.pairs))
        if not d.get('axis_valid'):
            info['reason']='no_current_periodic_line';return None,d,info
        if len(self.pairs)<PARAMS['minimum_pairs']:
            info['reason']='not_enough_past_pairs';return None,d,info
        offsets=np.asarray([p[1] for p in self.pairs]);offset=np.median(offsets,axis=0)
        scatter=np.median(np.linalg.norm(offsets-offset,axis=1))
        info['offset_MAD_px']=float(scatter)
        if scatter>PARAMS['maximum_offset_MAD_px']:
            info['reason']='unstable_past_center_offset';return None,d,info
        _,_,angle,extent=self.pairs[-1];line=d['line']
        if abs(np.cos(np.deg2rad(line['angle']-angle)))<PARAMS['minimum_axis_cosine']:
            info['reason']='axis_changed';return None,d,info
        ratio=line['major']/extent
        if not 1/PARAMS['maximum_extent_ratio']<=ratio<=PARAMS['maximum_extent_ratio']:
            info['reason']='extent_changed';return None,d,info
        center=np.asarray(line['center'])+offset
        if np.any(center<b[:2]) or np.any(center>=b[2:]):
            info['reason']='center_outside_search';return None,d,info
        recovered=dict(t=t,detected=True,center=center.tolist(),confidence=.125,low_confidence=True,
                       measurement_source='current_periodic_line',bbox=None,
                       estimated_body_box=False)
        info.update(used=True,reason='fresh_periodic_center',center=center.tolist(),
                    current_support_pixels=d['selected_pixel_count'])
        return recovered,d,info
