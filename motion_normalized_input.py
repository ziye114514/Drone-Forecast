"""Translation-normalized rotor evidence in cutoff-time camera coordinates.

Use independent past confirmed displacement whenever available. Re-estimating
velocity from every packet's rotating brightness centroid injects blade phase
into the alignment. Raw early-half proposals are only cold-start hypotheses;
the unchanged period filter still decides whether any target is detected.
"""
from collections import deque
import numpy as np
from moving_event_proposal import propose,activity_roi
from native_event_alignment import past_warp


class MotionNormalizedInput:
    def __init__(self):self.history=deque()
    def prepare(self,events,t,on,off):
        while self.history and t-self.history[0][0]>.16:self.history.popleft()
        past=[r for r in self.history if r[0]<=t-.033333+1e-7]
        if not past:
            e,meta=propose(events,t,on,off,active=True)
            return e,dict(meta,velocity_source='earlier_half_cold_start_proposal')
        roi=activity_roi(on,off)
        if roi is None:return None,dict(qualified=False,reason='no_current_activity_ROI',velocity_source='past_confirmed_motion')
        x,y,X,Y=roi;end=round(t*1e6)
        e=events[(events['t']>=end-33333)&(events['t']<end)&(events['x']>=x)&(events['x']<X)&(events['y']>=y)&(events['y']<Y)]
        source,v=past[-1]
        aligned,meta=past_warp(e,t,v,source)
        return aligned,dict(meta,qualified=True,velocity_source='past_confirmed_native_displacement',
            roi=roi,ROI_is_not_rotor_detection=True,source_events=len(e),warped_events=len(aligned),
            velocity_source_t=source,original_timestamps_polarities_retained=True)

    def observe(self,row):
        motion=row.get('motion') or {};estimate=motion.get('estimate') or {};dt=motion.get('dt_s')
        body=row.get('body') or {}
        if dt and motion.get('current_observation') and body.get('detected') and not body.get('center_provisional') and estimate.get('score',0)>0 and not estimate.get('peak_at_search_boundary') and estimate.get('current_displacement_qualified',True):
            v=np.array([estimate['dx'],estimate['dy']])/dt
            if np.isfinite(v).all():self.history.append((row['t'],v))
