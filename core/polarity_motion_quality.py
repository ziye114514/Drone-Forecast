"""Use ON/OFF displacement disagreement as appearance uncertainty.

The two channels observe the same rigid translation. Their difference is a
proxy for blade phase / appearance error, not a measured extra acceleration.
Keep the original joint estimate; do not average away or invent observations.
"""
import numpy as np
from rotor_support_front import estimate_region_translation

def qualify(previous,support,current,current_support,shift,**options):
    if shift is None or shift.get('peak_at_search_boundary'):return shift
    if (shift.get('polarity_motion_quality') or {}).get('available'):return shift
    zero=np.zeros_like(previous[0]);channel=[]
    for i in (0,1):
        result=estimate_region_translation((previous[i],zero),support,(current[i],zero),
            current_support=current_support,**options)
        channel.append(result)
    out=dict(shift);out['polarity_motion_quality']=dict(available=False,calibrated_error_covariance=False)
    if any(r is None or r.get('peak_at_search_boundary') for r in channel):return out
    delta=np.array([[r['dx'],r['dy']] for r in channel],float)
    if not np.isfinite(delta).all():return out
    half_difference=(delta[0]-delta[1])/2
    extra=np.outer(half_difference,half_difference)
    out['displacement_covariance_px2']=(np.asarray(out.get('displacement_covariance_px2',np.eye(2)*4.))+extra).tolist()
    out['polarity_motion_quality'].update(available=True,ON_displacement_px=delta[0].tolist(),OFF_displacement_px=delta[1].tolist(),
        disagreement_px=float(np.linalg.norm(delta[0]-delta[1])),added_covariance_px2=extra.tolist(),
        original_joint_displacement_preserved=True,no_pixel_count_division=True)
    return out
