"""One current inter-frame transform of filtered rotor evidence.

Shares the existing OpenCV LK/RANSAC and project similarity qualification.
No body image, NN, reconstructed intensity, RPM identity or future input.
"""
import cv2
import numpy as np
from live_local_flow import reused_similarity


def _image(counts,mask,bounds,local_contrast=False):
    x,y,X,Y=bounds
    activity=np.where(mask[y:Y,x:X],counts[0][y:Y,x:X]+counts[1][y:Y,x:X],0.)
    if local_contrast:
        from local_contrast_motion import normalize
        raw=cv2.GaussianBlur(np.log1p(activity).astype(np.float32),(5,5),1.2)
        descriptor=normalize(raw)
        return np.uint8(np.clip(128+32*descriptor,0,255))
    image=np.uint8(np.clip(32*np.log1p(activity),0,255))
    return cv2.GaussianBlur(image,(5,5),1.2)


def reference_leverage(points,center,mode):
    points=np.asarray(points,float);center=np.asarray(center,float);z=points-points.mean(0);d=center-points.mean(0)
    covariance=z.T@z/len(points)
    if mode=='plane_affine':value=float(d@np.linalg.solve(covariance,d))
    else:value=float(d@d/max(float(np.trace(covariance)),1e-12))
    return 1+max(value,0.)


def estimate_plane_motion(previous,old_mask,current,mask,center,propagate_reference_uncertainty=False,require_center_support=False,local_contrast=False,translation_only=False):
    empty=dict(qualified=False,reason='insufficient_filtered_rotor_support',new_body_detector=False,
        input_is_filtered_rotor_ONOFF=True,full_pose_measured=False)
    oy,ox=np.nonzero(old_mask);cy,cx=np.nonzero(mask)
    if min(len(ox),len(cx))<32:return None,empty
    lo=np.maximum(0,np.floor((np.minimum([ox.min(),oy.min()],[cx.min(),cy.min()])-112)/8)*8).astype(int)
    hi=np.minimum([1280,720],np.ceil((np.maximum([ox.max(),oy.max()],[cx.max(),cy.max()])+113)/8)*8).astype(int)
    bounds=(*lo,*hi);a=_image(previous,old_mask,bounds,local_contrast);b=_image(current,mask,bounds,local_contrast)
    support=np.uint8(old_mask[lo[1]:hi[1],lo[0]:hi[0]])*255
    points=cv2.goodFeaturesToTrack(a,80,.01,4,mask=support,blockSize=3)
    if points is None or len(points)<12:return None,dict(empty,reason='few_rotor_corners')
    options=dict(winSize=(21,21),maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4)
    q,forward,_=cv2.calcOpticalFlowPyrLK(a,b,points,None,**options)
    if q is None:return None,dict(empty,reason='forward_LK_failed')
    back,backward,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,**options)
    if back is None:return None,dict(empty,reason='backward_LK_failed')
    p=points.reshape(-1,2)+lo;q=q.reshape(-1,2)+lo;back=back.reshape(-1,2)+lo
    fb=np.linalg.norm(back-p,axis=1)
    valid=forward.ravel().astype(bool)&backward.ravel().astype(bool)&np.isfinite(q).all(1)&(fb<=1.5)
    inside=(q[:,0]>=0)&(q[:,0]<1280)&(q[:,1]>=0)&(q[:,1]<720)
    dilated=cv2.dilate(mask.astype(np.uint8),np.ones((5,5),np.uint8))
    indices=np.rint(q).astype(int);indices[:,0]=np.clip(indices[:,0],0,1279);indices[:,1]=np.clip(indices[:,1],0,719)
    valid&=inside&(dilated[indices[:,1],indices[:,0]]>0)
    p,q,fb=p[valid],q[valid],fb[valid]
    info=dict(empty,bidirectional_tracks=len(p),crop=list(map(int,bounds)))
    if len(p)<12:return None,dict(info,reason='few_bidirectional_rotor_tracks')
    eigen=np.linalg.eigvalsh(np.cov(p.T));mode='plane_affine' if eigen[0]>=16 and eigen[0]/eigen[-1]>=.025 else 'line_similarity'
    if translation_only:
        # A translation has no unobserved spatial shear to extrapolate from
        # remote background features to the drone. Retain the old2px and75%
        # consensus qualification; no video-specific model or noise tuning.
        mode='shared_translation';flow=q-p;delta=np.median(flow,axis=0)
        keep=np.linalg.norm(flow-delta,axis=1)<=2.
        if keep.sum()<12 or keep.mean()<.75:
            return None,dict(info,reason='weak_shared_translation_consensus',inliers=int(keep.sum()))
        delta=flow[keep].mean(0);residual=flow[keep]-delta
        matrix=np.c_[np.eye(2),delta]
        result=dict(valid=True,affine=matrix.tolist(),inliers=int(keep.sum()),inlier_fraction=float(keep.mean()),
            displacement_covariance_px2=(np.cov(residual.T)+np.eye(2)*max(1.,float(np.median(fb[keep]**2)))).tolist())
    elif mode=='plane_affine':
        matrix,inliers=cv2.estimateAffine2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=1000,confidence=.99,refineIters=10)
        if matrix is None:return None,dict(info,reason='plane_fit_failed')
        keep=inliers.ravel().astype(bool)
        residual=q[keep]-(p[keep]@matrix[:,:2].T+matrix[:,2])
        result=dict(valid=True,affine=matrix.tolist(),inliers=int(keep.sum()),inlier_fraction=float(keep.mean()),
            displacement_covariance_px2=(np.cov(residual.T)+np.eye(2)*max(1.,float(np.median(fb[keep]**2)))).tolist())
    else:
        result=reused_similarity(p,q,fb,center,center_support=True)
        if not result['valid']:return None,dict(info,reason=result['reason'])
        matrix=np.asarray(result['affine']);residual=q-(p@matrix[:,:2].T+matrix[:,2]);keep=np.linalg.norm(residual,axis=1)<=2.
    info.update(result,mode=mode,local_event_contrast_normalized=bool(local_contrast))
    if result['inliers']<12 or result['inlier_fraction']<.75:return None,dict(info,qualified=False,reason='weak_shared_motion_consensus')
    if require_center_support:
        # An affine map may transport an interpolated reference, not invent
        # motion at a centre outside all observed rotor tracks. A collapsed
        # line cannot constrain a general plane deformation.
        hull=cv2.convexHull(p[keep].astype(np.float32))
        inside=bool(cv2.pointPolygonTest(hull,tuple(np.asarray(center,float)),False)>=0.)
        info['center_in_current_track_hull']=inside
        if mode!='plane_affine' or not inside:
            return None,dict(info,qualified=False,reason='reference_not_supported_by_observed_plane')
    values=np.linalg.svd(matrix[:,:2],compute_uv=False)
    angle=np.rad2deg(np.arctan2(matrix[1,0]-matrix[0,1],matrix[0,0]+matrix[1,1]))
    extent=np.ptp(p[keep],axis=0);original=np.array([np.ptp(ox),np.ptp(oy)])
    long=int(np.argmax(original));coverage=float(extent[long]/max(original[long],1))
    if values.min()<.85 or values.max()>1.18 or abs(angle)>15 or coverage<.35:
        return None,dict(info,qualified=False,reason='shape_change_or_local_only_motion',span_fraction=coverage)
    center=np.asarray(center,float);delta=matrix[:,:2]@center+matrix[:,2]-center
    if not np.isfinite(delta).all() or max(abs(delta))>160:return None,dict(info,qualified=False,reason='nonlocal_displacement')
    info.update(qualified=True,reason='current_shared_rotor_transform',center_displacement=delta.tolist(),
        singular_values=values.tolist(),rotation_deg=float(angle),span_fraction=coverage)
    if propagate_reference_uncertainty:
        leverage=reference_leverage(p[keep],center,mode)
        Q=np.asarray(result['displacement_covariance_px2'],float);Q=(Q+Q.T)/2
        eig,U=np.linalg.eigh(Q);Q=(U*np.maximum(eig,4.))@U.T
        result['displacement_covariance_px2']=(Q*leverage).tolist()
        info.update(reference_prediction_leverage=leverage,displacement_covariance_px2=result['displacement_covariance_px2'],
            source_feature_centroid=np.mean(p[keep],axis=0).tolist(),reference_point_xy=center.tolist(),
            uncertainty_not_divided_by_feature_count=True,
            covariance_is_correlated_feature_proxy_not_calibrated_probability=True)
    shift=dict(dx=float(delta[0]),dy=float(delta[1]),score=result['inlier_fraction'],method='native_rotor_'+mode,
        search_px=160,region=[int(ox.min()),int(oy.min()),int(ox.max())+1,int(oy.max())+1],peak_at_search_boundary=False,
        shared_affine=matrix.tolist(),displacement_covariance_px2=result['displacement_covariance_px2'])
    return shift,info
