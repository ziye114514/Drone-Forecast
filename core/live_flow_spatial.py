"""Spatial support of a local similarity flow at the detector-box center.

Existing OpenCV convex hull/pointPolygonTest and ordinary least squares.
The leave-spatial-group spread is an engineering model-variation proxy, not
a calibrated confidence interval or a novel jackknife estimator.
"""
import copy
import numpy as np

DESIGN=dict(policies=['center-hull','spatial-covariance'],hull_margin_px=0.,
    spatial_groups='Four quadrants relative to the previous current box center',
    minimum_groups=3,minimum_subset_points=4,
    covariance='(G-1)/G * sum((leave-group center displacement - mean) outer product)',
    changes_existing_LK_thresholds=False,parameters_swept=False,
    true_center_or_rigid_body_not_proven=True,default_promoted=False)


def support(points,next_points,center,dt):
    import cv2
    p=np.asarray(points,float);q=np.asarray(next_points,float);center=np.asarray(center,float)
    hull=cv2.convexHull(p.astype(np.float32))
    signed=float(cv2.pointPolygonTest(hull,tuple(center),True))
    sectors=(p[:,0]>=center[0]).astype(int)+2*(p[:,1]>=center[1]).astype(int)
    counts=np.bincount(sectors,minlength=4);estimates=[];used=[]
    for group in np.flatnonzero(counts):
        keep=sectors!=group;a=p[keep]-center;b=q[keep]-center
        if len(a)<4:continue
        matrix=np.zeros((2*len(a),4));matrix[::2,0]=a[:,0];matrix[::2,1]=-a[:,1];matrix[::2,2]=1
        matrix[1::2,0]=a[:,1];matrix[1::2,1]=a[:,0];matrix[1::2,3]=1
        if np.linalg.matrix_rank(matrix)<4:continue
        params=np.linalg.lstsq(matrix,b.ravel(),rcond=None)[0]
        estimates.append(params[2:4]);used.append(int(group))
    covariance=None
    if len(estimates)>=3:
        values=np.asarray(estimates);delta=values-values.mean(axis=0)
        covariance=(len(values)-1)/len(values)*(delta.T@delta)/(dt*dt)
    return dict(center_in_hull=signed>=0.,signed_hull_distance_px=signed,quadrant_counts=counts.tolist(),
        hull=hull[:,0].tolist(),center=center.tolist(),leave_group_displacements=[x.tolist() for x in estimates],
        leave_group_ids=used,jackknife_groups=len(estimates),extra_velocity_covariance=None if covariance is None else covariance.tolist(),
        interpretation='Current matched feature support; neither true COM nor rotor identity.')


def observation(original,policy):
    if policy not in DESIGN['policies']:raise ValueError('Unknown spatial flow policy')
    if not original.get('valid'):return original
    result=copy.deepcopy(original);result['spatial_policy']=policy;s=result.get('spatial_support')
    if s is None:return dict(result,valid=False,reason='missing_spatial_support')
    if policy=='center-hull':
        if not s['center_in_hull']:return dict(result,valid=False,reason='center_outside_matched_feature_hull')
        return result
    if s['extra_velocity_covariance'] is None:return dict(result,valid=False,reason='few_spatial_leave_group_fits')
    result['velocity_covariance']=(np.asarray(result['velocity_covariance'])+s['extra_velocity_covariance']).tolist()
    result['reason']='qualified_spatial_model_variation_covariance'
    return result
