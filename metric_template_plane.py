"""Plane normal from the rigid five-circle scaffold, not separate ring votes.

An observed rotor circle supplies a plane metric at the first reliable four-
hub frame. Subsequent current rank6 three/four-hub affine updates transport
that metric. No equal-arm/square-drone, heading, real radius or mass is needed.
Two/line/old-template-only observations never acquire a full plane constraint.
"""
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from five_circle_model import ellipse_tensors


def log_shape(S):
    eig,U=np.linalg.eigh(S)
    if eig.min()<=1e-9:raise ValueError('Positive circle metric required')
    # A common radius/scale cancels from the projected plane normal.
    values=np.log(eig);values-=values.mean()
    return (U*values)@U.T


def exp_shape(L):
    eig,U=np.linalg.eigh((L+L.T)*.5);return (U*np.exp(eig))@U.T


class MetricTemplatePlane:
    def __init__(self):
        self.epoch=None;self.reference=None;self.metric_log=None;self.calibrations=0

    @staticmethod
    def current_metric(hubs,rotors):
        current=[r for r in rotors if r.get('current_observation',True) and not r.get('fov_clipped') and
            (r.get('mode')=='ellipse' or r.get('arc_center_qualified') or r.get('curvature_validated'))]
        if len(current)!=4:return None
        points=np.array([[r['cx'],r['cy']] for r in current]);cost=np.linalg.norm(hubs[:,None]-points[None],axis=2)
        ii,jj=linear_sum_assignment(cost)
        if any(cost[i,j]>max(4.,.15*current[j]['major']) for i,j in zip(ii,jj)):return None
        return np.mean([log_shape(S) for S in ellipse_tensors(current)],axis=0)

    def step(self,structure,geometry):
        out=copy.deepcopy(structure);epoch=structure.get('geometry_epoch')
        if self.epoch is not None and epoch!=self.epoch:self.reference=None;self.metric_log=None;self.calibrations=0
        self.epoch=epoch;continuous=structure.get('continuous') or {}
        fresh=bool(structure.get('mode') in ('four','three') and continuous.get('updated') and continuous.get('measurement_rank')==6)
        info=dict(applied=False,reason='no_current_rank6_five_circle_pose',heading_used=False,
            metric_radius_required=False,retained_template_is_not_current_pose=True)
        if not fresh:return out,info
        hubs=np.asarray(structure.get('model_centers'),float)
        if hubs.shape!=(4,2) or not np.isfinite(hubs).all():return out,dict(info,reason='missing_current_hubs')
        observed_metric=None
        if structure['mode']=='four':
            observed_metric=self.current_metric(hubs,geometry.get('rotors',[]))
        if self.reference is None:
            if observed_metric is None:return out,dict(info,reason='need_one_reliable_four_circle_metric')
            center=hubs.mean(0);scale=float(np.sqrt(np.mean(np.sum((hubs-center)**2,axis=1))))
            if scale<=1e-6:return out,dict(info,reason='collapsed_initial_scaffold')
            self.reference=np.c_[np.ones(4),(hubs-center)/scale];self.scale=scale
            self.metric_log=observed_metric;self.calibrations=1
        beta=np.linalg.lstsq(self.reference,hubs,rcond=None)[0];A=beta[1:].T/self.scale
        if not np.isfinite(A).all() or abs(np.linalg.det(A))<1e-8:
            return out,dict(info,reason='current_scaffold_degenerate_use_side_branch')
        S=A@exp_shape(self.metric_log)@A.T;eig,U=np.linalg.eigh(S)
        if eig.min()<=0:return out,dict(info,reason='current_metric_not_finite')
        ratio=float(np.sqrt(eig[0]/eig[1]));normal=U[:,0]*np.sqrt(max(0.,1-ratio**2))
        if normal[1]>0:normal=-normal
        if ratio<=.16:return out,dict(info,reason='metric_plane_is_line_use_existing_side_branch')
        out.update(axis_xy=normal.tolist(),axis_source='current_metric_five_circle_template',
            metric_plane_uses_current_hubs=True,metric_plane_is_full_3d_pose=False)
        info.update(applied=True,reason='current_rank6_metric_plane',ellipse_ratio=ratio,
            normal_xy=normal.tolist(),affine=A.tolist(),calibration_observations_before_current=self.calibrations,
            uncertainty_is_calibrated=False,hidden_motor_frequencies_measured=0)
        if observed_metric is not None:
            # Reconcile the fixed object metric only AFTER using the previous
            # calibration to predict this frame's plane. No future observation.
            current=exp_shape(observed_metric);inv=np.linalg.inv(A);back=log_shape(inv@current@inv.T)
            self.metric_log=(self.calibrations*self.metric_log+back)/(self.calibrations+1);self.calibrations+=1
        return out,info
