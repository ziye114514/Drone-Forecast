"""Use temporal seeds to group ORIGINAL outer arcs, never the cut-mask edges.

Seed centroids are not rotor centres. The existing conic/line fitter must
recover centres from actual contour samples, and existing uncertainty gates
decide whether a centre is observable. No target count of four is enforced.
"""
import cv2,numpy as np
from event_drone_forecast.geometry import fit_rotor_geometry,_distance


def longest_run(points,keep):
    if not np.any(keep):return points[:0]
    if np.all(keep):return points
    start=int(np.flatnonzero(~keep)[0]);order=np.roll(np.arange(len(points)),-start)
    idx=order[keep[order]];cuts=np.flatnonzero(np.diff(np.flatnonzero(keep[order]))>1)+1
    groups=np.split(idx,cuts);return points[max(groups,key=len)]


def split_original_arcs(count_mask,period_mask,max_major=500.):
    original=cv2.morphologyEx(count_mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    original=cv2.morphologyEx(original,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(original,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    rows=[];children=[];qualified_arcs=[]
    def qualified_center(fit):
        return bool(fit is not None and not fit.get('axis_only') and fit.get('mode')!='line' and
            fit.get('visible_arc_fraction',0)>=.48 and fit.get('support_fraction',0)>=.6 and
            fit.get('fit_condition',1e9)<=1500 and fit.get('center_sigma_px',1e9)<=3 and
            fit.get('axis_angle_sigma_deg',90)<=15)
    for parent_index,c in enumerate(contours):
        if len(c)<32 or cv2.contourArea(c)<40:continue
        x,y,w,h=cv2.boundingRect(c);parent=np.zeros((h,w),np.uint8)
        cv2.drawContours(parent,[c-[x,y]],-1,255,-1)
        seeds=period_mask[y:y+h,x:x+w].copy()
        seeds=cv2.morphologyEx(seeds,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        seeds=cv2.dilate(seeds,np.ones((3,3),np.uint8));seeds=cv2.bitwise_and(seeds,parent)
        n,labels,stats,centres=cv2.connectedComponentsWithStats(seeds)
        largest=max(stats[1:,4],default=0)
        ids=[j for j in range(1,n) if stats[j,4]>=max(30.,largest/(2.5**2))]
        info=dict(parent_index=parent_index,temporal_seed_count=len(ids),accepted=False,
            seed_centroids_are_hubs=False,original_boundary_points_only=True,children=[])
        if not 2<=len(ids)<=4:
            info['reason']='no_small_set_of_independent_temporal_seeds';rows.append(info);continue
        parent_fit=fit_rotor_geometry(c,max_major=max_major,allow_arc=True)
        info['same_fitter_without_temporal_split_qualified']=qualified_center(parent_fit)
        distance=np.array([cv2.distanceTransform(np.uint8(labels!=j),cv2.DIST_L2,5) for j in ids])
        points=c.reshape(-1,2);local=points-[x,y];d=distance[:,local[:,1],local[:,0]]
        owner=d.argmin(0);sorted_distance=np.sort(d,axis=0);margin=sorted_distance[1]-sorted_distance[0]
        fits=[];rejected=False
        for i,seed_id in enumerate(ids):
            arc=longest_run(points,(owner==i)&(margin>=1.))
            fit=fit_rotor_geometry(arc.reshape(-1,1,2),max_major=max_major,allow_arc=True) if len(arc)>=16 else None
            if fit is None:
                info['children'].append(dict(points=len(arc),qualified=False,reason='insufficient_original_arc'));rejected=True;continue
            qualified=qualified_center(fit)
            record={key:(v.tolist() if isinstance(v,np.ndarray) else v) for key,v in fit.items()}
            record.update(points=len(arc),qualified=qualified,temporal_seed_label=seed_id,
                original_density_contour_points=arc.tolist(),
                original_boundary_points_only=True,geometry_from_seed_centroid=False)
            info['children'].append(record);fits.append((fit,arc));rejected|=not qualified
            # Keep a good CURRENT arc even when its sibling is unobservable.
            # This is an alternative partial observation, not a successful pair
            # split or proof of a new physical motor/complete five-circle pose.
            if qualified and not info['same_fitter_without_temporal_split_qualified']:
                qualified_arcs.append(dict(record,parent_index=parent_index,
                    complete_pair_or_four_rotor_count_confirmed=False,
                    model_slot_identity_resolved=False,partial_observation_only=True))
        if not rejected and len(fits)==len(ids):
            hubs=np.array([f['center'] for f,a in fits]);major=np.array([f['major'] for f,a in fits])
            separation=np.linalg.norm(hubs[:,None]-hubs[None],axis=2)+np.eye(len(hubs))*1e6
            coherent=separation.min()>=max(12.,.35*np.median(major)) and major.max()/major.min()<=1.8
            # Real parent boundary must be explained; no manufactured closure.
            residual=[]
            for f,a in fits:
                p=np.r_[f['center'],f['major']/2,f['minor']/2,np.deg2rad(f['angle'])]
                residual.append(abs(_distance(points,p)))
            support=float(np.mean(np.min(residual,axis=0)<=2.))
            info['union_original_boundary_support']=support
            info['accepted']=bool(coherent and support>=.65)
        info['reason']='current_temporal_seeds_and_original_arcs' if info['accepted'] else 'arc_uncertainty_or_joint_geometry_rejected'
        if info['accepted']:children.extend(info['children'])
        rows.append(info)
    return dict(parents=rows,accepted_split_parents=sum(r['accepted'] for r in rows),children=children,
        new_current_qualified_centres=len(children),qualified_partial_arcs=qualified_arcs,
        new_qualified_partial_arcs=len(qualified_arcs),hidden_centres_fabricated=0)
