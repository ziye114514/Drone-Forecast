"""Choose center translation from an existing current similarity fit.

No new tracking/NN pass. A fit supported only on one side cannot transport an
unobserved center. The original median displacement is the exact fallback.
"""
import numpy as np


def choose(median,similarity,box):
    median=np.asarray(median,float)
    info=dict(applied=False,reason='current_center_not_rigidly_supported',
        median_displacement=median.tolist(),extra_LK_passes=0,new_detector_models=0)
    if not similarity.get('valid') or similarity.get('inliers',0)<12 or similarity.get('inlier_fraction',0)<.75:
        return median,info
    if not similarity.get('center_in_track_hull',False):return median,dict(info,reason='center_outside_current_inlier_hull')
    covariance=np.asarray(similarity.get('displacement_covariance_px2',np.eye(2)*1e9))
    delta=np.asarray(similarity.get('displacement'),float);box=np.asarray(box,float)
    if covariance.shape!=(2,2) or delta.shape!=(2,) or not np.isfinite(covariance).all() or not np.isfinite(delta).all():return median,info
    if np.linalg.eigvalsh(covariance).max()>9.:return median,dict(info,reason='rigid_residual_scatter_too_large')
    if np.linalg.norm(delta)>.4*max(box[2:]-box[:2]):return median,dict(info,reason='rigid_translation_too_large')
    moved=box+np.tile(delta,2)
    if moved[0]>=1280 or moved[1]>=720 or moved[2]<=0 or moved[3]<=0:return median,dict(info,reason='rigid_center_would_leave_sensor')
    return delta,dict(info,applied=True,reason='reuse_current_center_similarity',
        center_displacement=delta.tolist(),change_from_median_px=float(np.linalg.norm(delta-median)),
        reference_source=similarity.get('evaluation_center_source'),
        rotation_and_scale_not_treated_as_center_translation=True)
