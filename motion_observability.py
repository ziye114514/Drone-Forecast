"""Qualify motion separately from a small target's semantic confidence.

No detector confidence, body geometry, position eligibility, or rotor input is
upgraded. Three actual NN anchors and current coherent bidirectional tracks
only permit the already measured local displacement to be considered.
"""
from collections import deque
import numpy as np
from live_local_flow import box


def visible_motion_box(body):
    """A small truncated border need not erase an observed body displacement.

    This does NOT relax conic/FOV qualification. Return the original box so
    center-gauge and overlap checks keep the same coordinates. At least 90%
    of its area, its center, and >=12px in both axes must remain visible.
    """
    original=box(body)
    if original is not None:return original
    b=np.asarray(body.get('bbox',[]),float)
    if (not body.get('detected') or body.get('low_confidence') or b.shape!=(4,)
        or not np.isfinite(b).all() or np.any(b[2:]-b[:2]<12)):
        return None
    visible=np.r_[np.maximum(b[:2],[0.,0.]),np.minimum(b[2:],[1280.,720.])]
    size=visible[2:]-visible[:2]
    if np.any(size<12.) or np.prod(size)/np.prod(b[2:]-b[:2])<.9:
        return None
    c=np.asarray(body['center'])
    if not np.all((c>[0.,0.])&(c<[1280.,720.])):return None
    return b


class WeakSemanticMotion:
    def __init__(self,allow_spatial=False):
        self.anchors=deque();self.last=None
        self.allow_spatial=allow_spatial

    def observe(self,t,body,tracking):
        if self.last is not None and t-self.last>.5:
            self.anchors.clear()
        self.last=t
        b=box(dict(body,low_confidence=False))
        if b is None or body.get('confidence',0)<.25 or max(b[2:]-b[:2])>192.:
            return dict(qualified=False,reason='not_a_current_small_semantic_candidate')
        if body.get('measurement_source') in ('current_NN','current_KLT_from_delayed_NN'):
            stamp=float(body.get('nn_observed_t',t))
            if not self.anchors or stamp>self.anchors[-1]+1e-8:
                self.anchors.append(stamp)
        while self.anchors and t-self.anchors[0]>.35+1e-8:
            self.anchors.popleft()
        info=dict(qualified=False,reason='few_distinct_recent_NN_anchors',
            distinct_NN_anchors=len(self.anchors),body_confidence_unchanged=True,
            position_and_rotor_eligibility_unchanged=True)
        if len(self.anchors)<3 or self.anchors[-1]-self.anchors[0]<.16-1e-8 or t-self.anchors[-1]>.16+1e-8:
            return info
        similarity=tracking.get('similarity_motion') or {}
        spatial=similarity.get('spatial_support') or {}
        spatial_supported=bool(self.allow_spatial and spatial.get('center_in_hull') and
            np.count_nonzero(spatial.get('quadrant_counts',[]))>=3 and spatial.get('jackknife_groups',0)>=3 and
            spatial.get('extra_displacement_covariance_px2') is not None)
        minimum=8 if spatial_supported else 12
        if tracking.get('reason')!='current_image_translation' or tracking.get('points',0)<minimum:
            return dict(info,reason='few_current_bidirectional_tracks')
        if not similarity.get('valid') or similarity.get('inliers',0)<minimum or similarity.get('inlier_fraction',0)<.75:
            return dict(info,reason='current_similarity_not_supported')
        covariance=np.asarray(similarity.get('displacement_covariance_px2',np.eye(2)*1e9))
        if spatial_supported:covariance=covariance+np.asarray(spatial['extra_displacement_covariance_px2'])
        if covariance.shape!=(2,2) or not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).max()>9.:
            return dict(info,reason='current_displacement_scatter_too_large')
        return dict(info,qualified=True,reason='repeated_semantics_and_current_rigid_motion',
            current_similarity_inliers=similarity['inliers'],
            spatial_support_used=spatial_supported,
            displacement_variance_max_px2=float(np.linalg.eigvalsh(covariance).max()),
            observed_motion_is_not_measured_thrust=True)
