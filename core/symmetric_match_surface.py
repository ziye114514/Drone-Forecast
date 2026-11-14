"""One translation must explain both ordered native image comparisons.

Unlike averaging independently selected peaks, optimize the symmetric NCC
objective on the intersection of actually sampled displacements. No image
outside the sensor, covariance reduction, or trajectory prior is invented.
"""
import numpy as np
from native_peak_shape import assess


def combine(forward,backward):
    a=forward.pop('_matching_surface',None)
    b=backward.pop('_matching_surface',None) if backward else None
    r=dict(forward);info=dict(applied=False,uses_joint_cost_not_average_of_argmax=True,independent_observations=False)
    r['symmetric_surface_motion']=info
    if a is None or b is None:return r
    A=np.asarray(a['score']);B=np.asarray(b['score']);lo=np.asarray(a['minimum_xy'],int);blo=np.asarray(b['minimum_xy'],int)
    hi=lo+np.array(A.shape[::-1])-1;bhi=blo+np.array(B.shape[::-1])-1
    common_lo=np.maximum(lo,-bhi);common_hi=np.minimum(hi,-blo)
    if np.any(common_hi-common_lo<2):info['reason']='insufficient_common_sampled_domain';return r
    yy,xx=np.indices(tuple((common_hi-common_lo+1)[::-1]));xs=xx+common_lo[0];ys=yy+common_lo[1]
    aa=A[ys-lo[1],xs-lo[0]];bb=B[-ys-blo[1],-xs-blo[0]]
    joint=(aa+bb)/2.;index=np.unravel_index(int(np.argmax(joint)),joint.shape)
    if not np.isfinite(joint[index]) or joint[index]<=0:info['reason']='no_positive_common_match';return r
    value=common_lo+np.array(index[::-1],float)
    peak=assess(joint,index,float(r.get('search_px',40)))
    refinement=peak.get('joint_refinement_xy')
    if refinement is not None:value+=np.asarray(refinement)
    edge=any(index[i] in (0,joint.shape[i]-1) for i in (0,1))
    r.update(dx=float(value[0]),dy=float(value[1]),peak_at_search_boundary=bool(edge))
    info.update(applied=True,reason='joint_forward_reverse_native_NCC',integer_domain_xyxy=[*common_lo.tolist(),*common_hi.tolist()],
        forward_score=float(aa[index]),backward_score=float(bb[index]),joint_normalized_score=float(joint[index]),
        subpixel_refinement_xy=refinement,forward_displacement_xy=[forward['dx'],forward['dy']],
        reverse_displacement_xy=[backward['dx'],backward['dy']],sampled_common_boundary=bool(edge),covariance_reduced=False)
    return r
