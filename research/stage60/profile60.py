"""Density support check around a paired stripe, not a physical endpoint oracle."""
import cv2,numpy as np
PARAMS=dict(half_window_length=.8,low_threshold_multiplier=.5,across_support_fraction=.35,
            profile_close_bins=3,end_margin_bins=3,span_ratio=[.6,1.8],
            max_center_shift_fraction=.12,min_center_shift_px=2.,mirror_difference_max=.35)

def check(density,threshold,p):
    center=np.array(p['center']);length=p['major'];angle=np.deg2rad(p['angle']);u=np.array([np.cos(angle),np.sin(angle)]);n=np.array([-u[1],u[0]])
    along=np.arange(-np.ceil(length*.8),np.ceil(length*.8)+1)
    across=np.arange(-max(2,int(np.ceil(p['minor']*.6))),max(2,int(np.ceil(p['minor']*.6)))+1)
    q=center[None,None,:]+along[:,None,None]*u+across[None,:,None]*n
    h,w=density.shape;inside=(q[:,:,0]>=0)&(q[:,:,0]<w-1)&(q[:,:,1]>=0)&(q[:,:,1]<h-1)
    if not inside.all():return dict(usable=False,reason='profile_crosses_sensor_boundary')
    sampled=cv2.remap(density,q[:,:,0].astype(np.float32),q[:,:,1].astype(np.float32),cv2.INTER_LINEAR)
    profile=np.mean(sampled>=threshold*PARAMS['low_threshold_multiplier'],axis=1)
    mask=np.uint8(profile>=PARAMS['across_support_fraction'])[None,:]
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((1,PARAMS['profile_close_bins']),np.uint8))[0]>0
    mid=int(np.argmin(abs(along)))
    if not mask[mid]:return dict(usable=False,reason='no_supported_stripe_interior')
    lo=hi=mid
    while lo>0 and mask[lo-1]:lo-=1
    while hi<len(mask)-1 and mask[hi+1]:hi+=1
    if lo<PARAMS['end_margin_bins'] or hi>=len(mask)-PARAMS['end_margin_bins']:
        return dict(usable=False,reason='support_reaches_profile_window_edge')
    span=float(along[hi]-along[lo]);offset=float((along[hi]+along[lo])/2)
    if not PARAMS['span_ratio'][0]<=span/length<=PARAMS['span_ratio'][1]:return dict(usable=False,reason='support_length_inconsistent',span=span)
    shift_gate=max(PARAMS['min_center_shift_px'],PARAMS['max_center_shift_fraction']*length)
    if abs(offset)>shift_gate:return dict(usable=False,reason='one_sided_or_nonuniform_support',center_shift_px=abs(offset))
    mirror=float(np.mean(abs(profile[lo:hi+1]-profile[lo:hi+1][::-1])))
    if mirror>PARAMS['mirror_difference_max']:return dict(usable=False,reason='asymmetric_profile',mirror_difference=mirror)
    c=center+offset*u
    return dict(usable=True,reason='bounded_symmetric_density_support',center=c.tolist(),major=span,
       midpoint_shift_px=abs(offset),mirror_difference=mirror,profile_support=float(np.mean(profile[lo:hi+1])),
       endpoints=np.array([center+along[lo]*u,center+along[hi]*u]).tolist(),
       endpoint_semantics='Activity support ends, not independently verified physical blade tips.',verified_rotor_identity=False)

def checks():
    image=np.zeros((100,200),np.float32);image[48:53,50:151]=10
    p=dict(center=[100.,50.],major=100.,minor=4.,angle=0.)
    a=check(image,2.,p);assert a['usable'] and np.allclose(a['center'],[100,50])
    image[:,50:80]=0;b=check(image,2.,p);assert not b['usable'] and b['reason']=='one_sided_or_nonuniform_support'
    image=np.ones((100,200),np.float32)*10;c=check(image,2.,p);assert not c['usable'] and c['reason']=='support_reaches_profile_window_edge'
    return dict(symmetric_known_stripe_exact=True,one_sided_support_rejected=True,unbounded_support_rejected=True,body_stripe_not_semantically_identified=True)
