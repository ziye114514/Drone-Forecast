"""Select/reject current NN boxes; never publish a predicted box as detection.

Uses the project's existing6px center-noise and99% chi-square2D gate. A short
finite-difference search reference is association support only, not another
forecast/KF. Actual source timestamps handle variable inference cadence.
"""
from collections import deque
import numpy as np


class CurrentBoxGate:
    def __init__(self):self.history=deque();self.last=None

    def step(self,boxes,t,current_hint=None):
        if self.last is not None and t<=self.last:raise ValueError('Increasing NN source times required')
        self.last=t
        while self.history and t-self.history[0][0]>.5+1e-8:self.history.popleft()
        audit=dict(accepted=False,current_NN_only=True,coasting_is_detection=False,
                   confidence_changed=False,chi2_threshold=9.21034037197618)
        if not boxes:return [],dict(audit,reason='no_current_NN_candidates')
        selected=None
        if not self.history:
            selected=max(boxes,key=lambda b:b['confidence']);audit['reason']='new_current_NN_acquisition'
        else:
            stamp,box=self.history[-1];center=(box[:2]+box[2:])/2;gap=t-stamp
            velocities=[];intervals=[]
            for (ta,a),(tb,b) in zip(list(self.history)[:-1],list(self.history)[1:]):
                if 0<tb-ta<=.16:
                    velocities.append(((b[:2]+b[2:])-(a[:2]+a[2:]))/(2*(tb-ta)));intervals.append(tb-ta)
            if velocities:
                center=center+gap*np.median(velocities,axis=0)
                interval=float(np.median(intervals));variance=72.*(1+(gap/interval)**2)
            else:variance=72.+(.5*max(box[2:]-box[:2]))**2
            # Existing acceleration cap supplies bounded maneuver search room.
            variance+=(.5*3000.*gap*gap)**2
            if current_hint is not None:
                b=np.asarray(current_hint,float);center=(b[:2]+b[2:])/2;variance=max(variance,72.)
                audit['current_image_hint_used']=True
            candidates=[]
            for b in boxes:
                q=np.asarray(b['bbox']);p=(q[:2]+q[2:])/2;chi2=float(np.sum((p-center)**2)/variance)
                candidates.append((chi2,b))
            value,chosen=min(candidates,key=lambda x:x[0]);audit.update(best_chi2=value,association_reference=center.tolist(),search_variance_px2=variance)
            if value<=audit['chi2_threshold']:selected=chosen;audit['reason']='current_NN_inside_causal_search_gate'
            else:audit['reason']='current_NN_disagrees_with_recent_target'
        if selected is None:return [],audit
        b=np.asarray(selected['bbox'],float);self.history.append((t,b.copy()));audit['accepted']=True
        return [selected],audit
