"""Calibrate accumulated event-feature translation against fresh body anchors.

Rotor patterns can move coherently inside a body box. A low residual of the
feature fit does not prove that this apparent velocity is body translation.
The current detector/five-circle body anchors constrain a slow common drift;
there is no additional position filter and no head/tail or rotor ID input.
"""
from collections import deque
import numpy as np
from live_local_flow import box

class FlowAnchorGauge:
    def __init__(self):
        self.anchors=deque();self.integral=np.zeros(2);self.last_t=None;self.nn=None
        self.bias=None;self.bias_se=None;self.last_update=None;self.last_box=None
    def step(self,t,body,tracking,observation):
        info=dict(applied=False,reason='anchor_gauge_warmup',no_new_position_filter=True)
        dt=None if self.last_t is None else t-self.last_t;self.last_t=t
        measured=bool(dt is not None and 0.<dt<=.08+1e-8 and
            tracking.get('reason')=='current_image_translation' and tracking.get('points',0)>=8)
        if not measured:
            self.anchors.clear();self.integral[:]=0.;self.bias=None;self.bias_se=None;self.last_update=None
        else:
            similarity=tracking.get('similarity_motion') or {}
            delta=np.asarray(similarity['displacement'] if similarity.get('valid') else tracking['delta'])
            self.integral+=delta
        b=box(body);stamp=body.get('nn_observed_t')
        new=bool(b is not None and body.get('confidence',0)>=.45 and stamp is not None and stamp!=self.nn
            and body.get('measurement_source') in ('current_NN','current_KLT_from_delayed_NN'))
        if new:
            self.nn=stamp
            # Large visibility/scale changes can relocate a box center. They
            # establish a new positional gauge, not a drift observation.
            if self.last_box is not None:
                size=b[2:]-b[:2];old=self.last_box[2:]-self.last_box[:2]
                if np.max(abs(size-old))>12.:
                    self.anchors.clear();self.bias=None;self.bias_se=None;self.last_update=None
            self.last_box=b.copy()
            self.anchors.append((t,np.asarray(body['center'])-self.integral))
        while self.anchors and t-self.anchors[0][0]>.5+1e-8:self.anchors.popleft()
        if new and len(self.anchors)>=3:
            tt=np.array([r[0]-t for r in self.anchors]);yy=np.array([r[1] for r in self.anchors]);span=float(np.ptp(tt))
            if span>=.16-1e-8:
                A=np.column_stack((np.ones(len(tt)),tt));beta=np.linalg.lstsq(A,yy,rcond=None)[0]
                error=A@beta-yy;mad=1.4826*np.median(abs(error-np.median(error,axis=0)),axis=0)
                noise=np.maximum(6.,mad)
                se=noise/np.sqrt(max(float(np.sum((tt-tt.mean())**2)),1e-12))
                if np.max(np.linalg.norm(error,axis=1))<=12.:
                    self.bias=-beta[1];self.bias_se=se;self.last_update=t
                    info.update(anchor_rows=len(tt),anchor_span_s=span,fit_residual_px=error.tolist())
        if self.bias is None or self.last_update is None or t-self.last_update>.12+1e-8:
            return observation,info
        if not observation.get('valid'):return observation,dict(info,reason='no_current_translation_for_gauge')
        reliability=np.maximum(0.,1.-self.bias_se**2/(self.bias**2+1e-12))
        correction=-reliability*self.bias
        R=np.asarray(observation['velocity_covariance'])+np.diag(self.bias_se**2)
        value=dict(observation,velocity=(np.asarray(observation['velocity'])+correction).tolist(),
            velocity_covariance=R.tolist(),uncorrected_velocity=observation['velocity'],
            gauge_correction_px_s=correction.tolist())
        return value,dict(info,applied=bool(np.any(reliability>0.)),reason='matured_body_anchor_translation_gauge',
            estimated_bias_px_s=self.bias.tolist(),bias_standard_error_proxy=self.bias_se.tolist(),
            correction_px_s=correction.tolist(),last_update_age_s=t-self.last_update,
            raw_feature_tracks_changed=False,heading_used=False,rotor_ids_used=False)
