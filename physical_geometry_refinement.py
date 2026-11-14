"""Optional cumulative rotor geometry: event-time alignment and metric relock.

Uses one past velocity, original pixel support, original current raw events,
and a persistent single-target five-circle metric. No trajectory labels,
recording-specific branches, inferred motor frequencies, or NN inputs.
"""
from collections import deque
import numpy as np
from fast_structure_front import roi_bounds,count_pair
from native_event_alignment import past_warp
from metric_model_relock import MetricModelRelock


class PhysicalGeometryRefinement:
    def __init__(self,predictor):
        self.predictor=predictor;self.geometry=predictor.front.geometry
        self.original_step=self.geometry.step;self.history=deque();self.metric=MetricModelRelock()
        self.events=None;self.audit={};self.geometry.step=self._step

    def _step(self,e,t,body,q,velocity,**kwargs):
        counts=self.geometry.count_current;meta=dict(applied=False,reason='no_past_translation_or_current_support')
        prior=[r for r in self.history if r[0]<=t-.020+1e-7]
        if body.get('detected') and counts is not None and self.events is not None and prior:
            bounds=roi_bounds(body['bbox']);on,off=counts(e,bounds,t,self.geometry.selector)
            x,y,X,Y=bounds;raw=self.events
            keep=(raw['t']>=round(t*1e6)-20000)&(raw['t']<round(t*1e6))&(raw['x']>=x)&(raw['x']<X)&(raw['y']>=y)&(raw['y']<Y)
            raw=raw[keep];ix=raw['x'].astype(int)-x;iy=raw['y'].astype(int)-y
            support=((on+off)[iy,ix]>0);selected=raw[support]
            aligned,meta=past_warp(selected,t,prior[-1][1],prior[-1][0],window_us=20000)
            if meta['applied']:
                on,off=count_pair(aligned,bounds,t,cpp_histogram=True)
                def aligned_counts(_e,b,_t,_selector):
                    if not np.array_equal(bounds,b):raise ValueError('Current geometry ROI changed within one fit')
                    return on,off
                self.geometry.count_current=aligned_counts
        try:structure,extra,diag,last=self.original_step(e,t,body,q,velocity,**kwargs)
        finally:self.geometry.count_current=counts
        refined,metric=self.metric.step(t,body,structure,diag)
        self.audit=dict(event_time_alignment=meta,current_intrinsic_metric_pose=metric)
        return refined,extra,diag,last

    def step(self,events,t):
        while self.history and t-self.history[0][0]>.16:self.history.popleft()
        self.events=events
        try:r,frame=self.predictor.step(events,t)
        finally:self.events=None
        motion=r['motion'];estimate=motion.get('estimate') or {};dt=motion.get('dt_s')
        if dt and motion.get('current_observation') and estimate.get('score',0)>0 and not estimate.get('peak_at_search_boundary') and estimate.get('current_displacement_qualified',True):
            self.history.append((t,np.array([estimate['dx'],estimate['dy']])/dt))
        r['physical_geometry_refinement']=self.audit
        return r,frame
